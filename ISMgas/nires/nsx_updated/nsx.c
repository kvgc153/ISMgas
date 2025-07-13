/* nsx.c                      tab  November 2013 and March,April,May 2017 */

/* Process NIRES raw data files. */


/* TEST command  (in /home/tb/nsx/) :
   nsx arcsum.fits xsp=40,60 xbk=20,40 xbk=60,80
# nsx (Keck NIRES redux).
setenv NSXDIR '/home/tb/nsx/'
setenv NSXOUT '/home/tb/nsxout/'
*/



/* TODO: (to do future)  Fri May  5 14:22:24 PDT 2017 .and. Wed Apr 11 14:12:37 PDT 2018

  -- flat field: need to create (leveled) flat field image-- should improve background fitting..
     (NOTE: dome flat may not represent response well enough-- use science exposures?)
     (NOTE: need to do tests to see how well pixel-to-pixel flat works)

  -- Atmospheric absorption correction and flux calibration and combining orders..

  -- Rebinning problem.. the sky lines have ragged edges in the corrected images.. I need
     to model the lines (and object profile?) to do a better job at rebinning...

  -- DELIVER NEW VERSION (Sep2018)..

  -- what about information about centroid vs. trace offset in the -sp?.tbl files?
     (so we can tell how well the program trace w/DAR matched the actual data)..

  -- WHEN I have more star exposures are various airmass values:
     Investigate effects of position angle on the change of position of object
     along slit as a function of wavelength, and how that affects the trace..

  -- Allow echelle order dependent background and object regions.. (see  s180304_0046.fits ).

  -- Check new SlitOffset computation.. ?

  -- Function for coadding data..

  -- Function for auto-pipeline (select and extract objects)..

  -- interpolate over bad pixels for object and background spectrum extraction.

  -- PGPLOT plotting program (separate program).

  x- check slant solution in newer (180304) data.. (note: looks about the same)
  x- apply mask image for background fitting (ignore hot/low pixels, CRs, ..) (note: I just interpolate)
  x- mask image: create hot/low pixel map (mask image) from darks and flats..
  x- What is real pixel scale (use latest star)..  (changed august 2018 -tab)
  x- How do I detect and mark CRs?
  x- Calibrate arcseconds per pixel scale on all orders using star exposures.
  x- Recalibrate slant functions using sky lines after first light (see: nsx_calslant() )..
  x- Recalibrate wavelength scale using sky lines after first light...
  x- Can I improve background fitting?
  x- Should I create a straightened image (no echelle curve, no slant) for user info?
*/


/* NOTE: Fri Feb 16 15:25:04 PST 2018
   I estimate the eperdn=1.0 (by subtracting two flats) if I assume the HWHM is the error
   BUT, the gaussian 1 sigma is actually smaller than the HWHM by 2ln2 (1.17741)..  this
   implies an eperdn of 1.3 .. but I will use 1.0 for now since that is what the header says
   and there may be some unusual features to the NIRES noise distribution.. 
   ALSO NOTE: the subtracted flats shows features that are not expected from a pure
   gaussian noise distribution.. I left this in and computed HWHM from 17x17 boxes on flat
   subtracted image.
*/

/*
 Order 3 : 2.46 to 1.88 (K band: 2.0-2.4)
 Order 4 : 1.85 to 1.42 (H band: 1.5-1.8)
 Order 5 : 1.48 to 1.13 (J band: 1.1-1.4)
 Order 6 : 1.23 to 0.95 (z/J band)
 Order 7 : 1.06 to 0.92 (z band:~0.82-1.03)
                        (i band:~0.69-0.84)
*/




#include "/home/keerthi/work/cfitsio/fitsio.h"
#include "nsx.h"
#include "mua.h"
#include "cfua.h"
#include "galactic.h"


#define TESTPROCENT 0

#define Check_AVP_ASPP 0

#define Change_ASPP_in_AVP 0

#define nc_Nominal 2048
#define nr_Nominal 1024


#define TRACE_TEST_ARCSEC -999. 
/* 
#define TRACE_TEST_ARCSEC 9.30
#define TRACE_TEST_ARCSEC 7.236
#define TRACE_TEST_ARCSEC 5.223
*/


#define ATMABS_CORRECTION 0

#define VEGA_FLUX_CALIBRATION 0

#define CONSTRUCT_ATMABS 0

#define CONSTRUCT_ATMABS_DAT_TBL 0

#define NSXVERSION "2018 August 31"

#define PIXFWHM 2.2

#define ARCSEC_PER_PIXEL 0.150

/* OLD Value, before August 2018 .. 
#define ARCSEC_PER_PIXEL 0.123
*/


/* Measured arcseconds per pixel from  s180304_0047 .. s180304_0054 data
   is Median ASPP = 0.14987   Average=0.1499 (96 points) with rms=0.0052
   using nso=3,4,5,6 (not 7).. and using nsx_TestProCent() on 31jul2018 ..

   Measured arcseconds per pixel from  s171214_0017 and s171214_0018 data
   is  Median ASPP = 0.14797  Average ASPP = 0.14804   rms= 0.003 .. 
   BUT one of these was near the edge of slit, so perhaps not that accurate..
 
   Original: ARCSEC_PER_PIXEL 0.123 (before July 2018) from web docs..

   Adopt new value of 0.150 .. -tab31jul2018
*/


#define MASKSCALE 1

#define DOBACKSUB 0

#define CREATE_OFFIMG 3

#define SOP1CALIB 0    /* one time calibration from SLT to SOP .. */


/* #define MINOBJSIGS 5.0 */
#define MINOBJSIGS 1.0   /* Minimum sigmas for a profile object to get automatically extracted. */



/* . . . Guide to Structures and Parameters . . .

SLT : obsolete, used an image to define slant of skylines across slit..

NCP : used to define edges of slit (usu. lower edge)..

WSC : used to define wavelength calibration polynomials (pre-sky shift).

AVP : arcsec-vs-pixel, defines curvatures of an star within the slit for
      all echelle orders.. 

SlitOffset : a global variable that defines offset of a slit during a 
    given exposure (usu. the same during an observing run).  This is determined
    for each exposure and applied to AVP calculation.

SOP : Slant-Offset-Polynomials, defines slant of a skyline across the slit.
      Lots of polynomials for each column and echelle order.

SOPI: Inverse, used to compute row position of a slant offset.

Note: 'SOP1' are the initial slant-offset-polynomials (I think from arclines)..
Note: 'SOP2' are the corrections to SOP1 (I think from skylines)..

*/


/* Number of points in the mkts_* files in nsx/cal  */
#define NUMMKT 80500  

/* Mauna Kea (atmospheric) Transmission data */
/* Number of points for this structure is always NUMMKT . */
struct MKTstructure {
  double tran[5];     /* transmission values (index = nso-3). */
};
typedef struct MKTstructure MKTtype;


/* Mauna Kea (atmospheric) Transmission data (TEMPORARY) */
/* Number of points for this structure is always NUMMKT . */
struct MKTTstructure {
  double wave;        /* wavelength in Angstroms */
  double tran[5];     /* transmission values (index = nso-3). */
};
typedef struct MKTTstructure MKTTtype;


/* Slant calibration data. */
struct SLTstructure {
  int   nc;            /* number of columns */
  float *image;        /* image data */
};
typedef struct SLTstructure SLTtype;


/* Wavelength Scale Calibration for each echelle order. */
struct WSCstructure {
  int    order;
  double xoff;          /* polynomial solution for wave = func(column) */
  double coef[4];
  int    orderinv;
  double xoffinv;       /* polynomial solution for column = func(wave) */
  double coefinv[4];
};
typedef struct WSCstructure WSCtype;


/* NIRES Calibration Parameter for each echelle order. */
struct NCPstructure {
  int    porda;
  double xoffa;         /* polynomial solution for lower edge. */
  double coefa[9];
  int    pordb;
  double xoffb;         /* polynomial solution for upper edge. */
  double coefb[9];
};
typedef struct NCPstructure NCPtype;


/* Spectral extraction data 'SPX[nso]'. */
#define MAXasp 10
#define MAXabk 60
#define MAXPRO 300
#define MAXSP 3000
struct SPXstructure  {
  int nsp;                           /* SPX[0]: number of background regions. */
  double asp1[MAXasp],asp2[MAXasp];  /* SPX[0]: Object window boundaries (as pixels for each order). */
  int nbk;                           /* SPX[0]: number of background regions. */
  double abk1[MAXabk],abk2[MAXabk];  /* SPX[0]: Background window(s) boundaries (as arcseconds). */
  double pflx[MAXasp];       /* SPX[0]: Profile object flux. */
  double sigs[MAXasp];       /* SPX[0]: Profile object sigmas. */
  double peak[MAXasp];       /* SPX[0]: Profile object peak. */
  double pro_cen;            /* Centroid of main object in arcseconds in each order. */
  int    numpro;             /* Number of points in profile, i.e. nsx_minwith(). */
  int    pro_apn;            /* Number of elements in pro_apx[] and pro_apymed[] and pro_apyave[] . */
  double pro_apx[MAXPRO];    /* Arcsecond profile (arcseconds) (SPX[0] is average). */
  double pro_apymed[MAXPRO]; /* Arcsecond profile (median)  (SPX[0] is sum). */
  double pro_apyave[MAXPRO]; /* Arcsecond profile (average)  (SPX[0] is sum). */
  int    numsp;              /* Number of points in spectrum. */
  double spobj[MAXSP];       /* object spectrum sum of counts per second in object window (per column). */
  double sperr[MAXSP];       /* object spectrum error. */
  double spbck[MAXSP];       /* background spectrum per second (average over pixels in background window(s)). */
  double spsky[MAXSP];       /* sky spectrum per second (average over pixels in object window). */
  double spwav[MAXSP];       /* wavelength in Angstroms. */
  double spdsp[MAXSP];       /* dispersion: angstroms/pixel. */
  double sprow[MAXSP];       /* image row value of object spectrum. */
  double spatm[MAXSP];       /* atmospheric transmission. */
  double spoac[MAXSP];       /* object with atmospheric correction (spectrum divided by atmospheric transmission). */
  double speac[MAXSP];       /* error with atmospheric correction (error divided by atmospheric transmission). */
};
typedef struct SPXstructure SPXtype;



/* IMAGE data. */
struct IMGstructure {
  int    nc,nr;         /* number of columns and rows. */
  char   file[100];     /* filename */
  char   root[100];     /* filename without preceding directory or .fits extension */
  char   utshut[40];    /* UT shutter open start time */
  char   object[60];    /* Object name. */
  double jd;            /* Julian Day. */
  double ra,dec;        /* RA and DEC J2000 */
  double airmass;       /* Airmass from header. */
  double ha,az,el;      /* Hour angle, azimuth, elevation (degrees) */
  double parang;        /* Parallactic angle (degrees) (position angle) */
  double rotposn;       /* Rotator User Position (angle in degrees) */
  double exptime;       /* Exposure (integration) time in seconds (ITIME) */
  float  *image;        /* image data */
  float  *varimg;       /* variance image data */
  float  *clnimg;       /* image data (cleaned, removed CRs or hot pixels) */
  float  *corimg;       /* corrected image data (unslant and uncurve) */
  float  *corimgNFD;    /* corrected image data with No Flat Division (use for variance). */
  float  *bckimg;       /* background fit image data (corrected) */
  int    X;             /* does image exist?  1 or 0 */
};
typedef struct IMGstructure IMGtype;


/* AVP : Arcsec vs. Pixel position (row offset from lower edge) */
/* forward is: arcsec = func(rowoff)  and inverse is rowoff = func(arcsec).. */
/* 'rowoff' is the real row position minus the edge position at given nso,icol.. */
struct AVPstructure {
  double xoff[MAXSP];        /* xoff value for each column of AVP[nso] */
  double coef[MAXSP][3];     /* coeffecients (order=2) for each column of AVP[nso] */
  double xoffinv[MAXSP];     /* inverse xoff value for each column of AVP[nso] */
  double coefinv[MAXSP][3];  /* inverse coeffecients (order=2) for each column of AVP[nso] */
};
typedef struct AVPstructure AVPtype;


/* SOP : Slant Offset Polynomials (for a given nso at a given column). */
/* forward is: (pixel slant offset) = func(rowoff) .. */
struct SOPstructure {
  double xoff[MAXSP];        /* xoff value for each column of SOP[nso] */
  double coef[MAXSP][3];     /* coeffecients (order=2) for each column of SOP[nso] */
};
typedef struct SOPstructure SOPtype;


/* SOPI : Inverse Slant Offset Polynomials (for a given nso at a given column). */
/* inverse is rowoff = func(pixel slant offset).. */
struct SOPIstructure {
  double xoffinv[MAXSP];     /* inverse xoff value for each column of SOP[nso] */
  double coefinv[MAXSP][4];  /* inverse coeffecients (order=2) for each column of SOP[nso] */
};
typedef struct SOPIstructure SOPItype;



/* NIRES LiSting of exposures. */
#define MAXNLS 900
struct NLSstructure {
  int num;
  char xtype[9];
  char root[30];
  double ra,dec,air,jd,exp;
  double EffTemp;
};
typedef struct NLSstructure NLStype;


/* Tycho Star Catalog */
#define MAXTYC 2701000
struct TYCstructure {
  double mean_ra,mean_dec;
  float bt_mag,vt_mag;
  float bt_mag_error,vt_mag_error;
};
typedef struct TYCstructure TYCtype;




#define VERB 0
#define READ_DARKIMG 0


#define CALSLANT 0

#define CALCURVE  0
#define CALCURVE2 0


/* ORIGINAL:
#define EPERDN   4.49   /x default is 3.8 in cookbook x/
#define ROVDN    2.51   /x implies 11.3 electrons: cookbook says 10e(CDS), 5e(8 samples), 3.5e(16 samples) x/
#define RAWOFF  10.00
*/

/* Estimate from NIRES flats (feb2018 -tab). */
#define EPERDN   1.00

/* THIS IS FROM TripleSpec ? */
#define ROVDN    4.00
#define RAWOFF -10.00


/* Differential Atmospheric Refraction reference wavelength for NIRES. */
#define DAR_refwave 12000.

/* Global variables. */
FILE *logfu = NULL;
int verbose = 0;
double SlitOffset = 0.;
double N2_refwave = 0.;
NCPtype NCP[9];
WSCtype WSC[9];





/* ----------------------------------------------------------------------
 Find the nearest element in an array to the given value.
 This version uses a binary search for speed.
 The array must be sorted for this to work.
   Input: xx (given value).
   Input: narr (number of elements in array).
   Input: arr (the array to be searched).
 Array MUST be sorted, low values to high values.
*/
/*@@*/
int cneari_bs(double xx, int narr, double arr[])
{
/**/
int ii,kk,i1,i2;
/**/
/* Binary search. */
if (narr <= 1) return(0);
ii=0;
i1=0;
i2=narr-1;
while ((i2-i1) > 1) {
  ii = (i1+i2)/2;
  if (arr[ii] > xx) { i2=ii; } else { i1=ii; }
}
if (i2 <= i1) { fprintf(stderr,"***ERROR: cneari_bs: this should not happen.\n"); }
if ( ABS((arr[i1] - xx)) < ABS((arr[i2] - xx)) ) { kk=i1; } else { kk=i2; }
return(kk);
}



/* ----------------------------------------------------------------------
  THIS VERSION WORKS WITH 4 BYTE REALS (float).
  Find the nearest element in an array to the given value.
  This version uses a binary search for speed.
  The array must be sorted for this to work.
    Input: xx (given value).
    Input: narr (number of elements in array).
    Input: arr (the array to be searched).
  Array MUST be sorted, low values to high values.
*/
/*@@*/
int cneari_bs4(float xx, int narr, float arr[])
{
/**/
int ii,kk,i1,i2;
/**/
/* Binary search. */
if (narr <= 1) return(0);
ii=0;
i1=0;
i2=narr-1;
while ((i2-i1) > 1) {
  ii = (i1+i2)/2;
  if (arr[ii] > xx) { i2=ii; } else { i1=ii; }
}
if (i2 <= i1) { fprintf(stderr,"***ERROR: cneari_bs4: this should not happen.\n"); }
if ( ABS((arr[i1] - xx)) < ABS((arr[i2] - xx)) ) { kk=i1; } else { kk=i2; }
return(kk);
}







/* ----------------------------------------------------------------------
  Return the value of the black body function in ergs/sec/cm^2/Angstrom
  (i.e. energy per unit wavelength):
      flux(a,T,x) = a * ( (h*c*c)/x^5 ) / ( exp((h*c)/(k*T*x)) - 1 )
  Input:  a    : Arbitrary scale factor.
          T    : Temperature in Kelvin.
          x    : Wavelength in Angstroms.
*/
/*@@*/
double nsx_BlackBodyAT( double a, double T, double x )
{
/**/
  const double c = 2.997925e+18 ;  /*  Ang/sec      */
  const double k = 1.380662e-16 ;  /*  ergs/Kelvin  */
  const double h = 6.626176e-27 ;  /*  ergs*sec     */
  double p1 = h*c*c;
  double p2 = h*c/k;
/**/
  double r;
/**/
  r = ( a * p1  / (x*x*x*x*x) ) / ( exp(p2/(T*x)) - 1. ) ;
  return(r);
}

/* ----------------------------------------------------------------------
  Given a temperature and a weighted spectrum, return the best possible fit
  for the value "a" using:
     flux(a,T,x) = a * ( (h*c*c)/x^5 ) / ( exp((h*c)/(k*T*x)) - 1 )
  Use:
     a = SUM { Wi * Yi * flux(1,T,Xi) } / SUM { Wi * flux(1,T,Xi)^2 }
  Input: nn    : number of points in spectrum.
         xx[]  : wavelengths in Angstroms.
         yy[]  : flux in ergs/sec/cm^2/Angstroms.
         ww[]  : weight values.
         T     : Temperature in Kelvin.
*/
/*@@*/
double nsx_SolveBlackBodyA( int nn, float xx[], float yy[], float ww[], double T )
{
/**/
double a,sum1,sum2,x,y,w,flux;
int ii;
/**/
a   =1.;
sum1=0.;
sum2=0.;
for(ii=0; ii<nn; ++ii) {
  x = xx[ii];
  y = yy[ii];
  w = ww[ii];
  flux = nsx_BlackBodyAT(a,T,x);
  sum1 = sum1 + ( w * y * flux );
  sum2 = sum2 + ( w * flux * flux );
}
if (sum2 > 0.) { return(sum1/sum2); } else { return(0.); }
}

/* ----------------------------------------------------------------------
  Given a spectrum, and black-body fit parameters, return the least-squares
  value.
  Input: nn    : number of points in spectrum.
         xx[]  : wavelengths in Angstroms.
         yy[]  : flux in ergs/sec/cm^2/Angstroms.
         ww[]  : weight values.
  Input: T     : Temperature in Kelvin.
         a     : Temperature in Kelvin.
*/
/*@@*/
double nsx_LeastSquaresBlackBody( int nn, float xx[], float yy[], float ww[], 
                               double T, double a )
{
/**/
double sum,x,y,w,r;
int ii;
/**/
sum = 0.;
for (ii=0; ii<nn; ++ii) {
  x = xx[ii];
  y = yy[ii];
  w = ww[ii];
  r = (nsx_BlackBodyAT(a,T,x) - y);  
  sum = sum + (w * r * r);
}
return(sum);
}

/* ----------------------------------------------------------------------
  Given a spectrum, estimate the black-body temperature "T" and the scale
  parameter "a", assuming the flux is in ergs/sec/cm^2/Angstrom
  (i.e. energy per unit wavelength):
       flux(a,T,x) = a * ( (h*c*c)/x^5 ) / ( exp((h*c)/(k*T*x)) - 1 )
  Input: nn    : number of points in spectrum.
         xx[]  : wavelengths in Angstroms.
         yy[]  : flux in ergs/sec/cm^2/Angstroms.
         ww[]  : weight values.
 Output: T     : Temperature in Kelvin.
         a     : Temperature in Kelvin.
*/
/*@@*/
void nsx_FitBlackBody( int nn, float xx[], float yy[], float ww[], 
                       double *T, double *a )
{
/**/
double tt,aa,lsf,lo_lsf,lo_aa,lo_tt;
/**/
/* Set up. */
lo_tt = 6000.;
lo_aa = nsx_SolveBlackBodyA( nn, xx, yy, ww, lo_tt );
lo_lsf= nsx_LeastSquaresBlackBody( nn, xx, yy, ww, lo_tt, lo_aa );
/* Find the best temperature. */
for (tt=2000.; tt<9000.; tt=tt+200.) {
  aa = nsx_SolveBlackBodyA( nn, xx, yy, ww, tt );
  lsf= nsx_LeastSquaresBlackBody( nn, xx, yy, ww, tt, aa );
  if (lsf < lo_lsf) { lo_lsf=lsf; lo_aa=aa; lo_tt=tt; }
}
for (tt=9000.; tt<21000.; tt=tt+500.) {
  aa = nsx_SolveBlackBodyA( nn, xx, yy, ww, tt );
  lsf= nsx_LeastSquaresBlackBody( nn, xx, yy, ww, tt, aa );
  if (lsf < lo_lsf) { lo_lsf=lsf; lo_aa=aa; lo_tt=tt; }
}
*T = lo_tt;
*a = lo_aa;
return;
}





/* ----------------------------------------------------------------------
    Find the nearest element in an array to the given value.
       Input: xx (given value).
       Input: narr (number of elements in array).
       Input: arr (the array to be searched).
    Array need not be sorted.
*/
int nsx_cneari(double xx, int narr, double arr[])
{
/**/
int ii,bestii;
double lo,diff;
/**/
bestii=0;
lo=ABS((xx - arr[0]));
for (ii=0; ii<narr; ++ii) {
  diff = ABS((xx - arr[ii]));
  if (diff < lo) { lo=diff; bestii=ii; }
}
return(bestii);
}


/* ----------------------------------------------------------------------
 Append a slash if needed.
*/
void nsx_append_slash( char wrd[] )
{
int ii;
ii = clc(wrd);
if (ii >= 0) { if (wrd[ii] != '/') { wrd[(ii+1)]='/';   wrd[(ii+2)]='\0'; } }
return;
}


/* ----------------------------------------------------------------------
  Load forward and inverse wavelength calibration polynomials for each nso.
*/
void nsx_load_WSC( char nsxdir[] )
{
/**/
int nso;
char wrd[200];
char line[200];
FILE *infu;
/**/
for (nso=3; nso<=7; ++nso) {
  sprintf(wrd,"%scal/wave_nsx_%d.poly",nsxdir,nso);
  infu = fopen_read(wrd);
  fgetline(line,infu);
  fgetline(line,infu);
  WSC[nso].order   = 3;
  WSC[nso].xoff    = GLV(line,1);
  WSC[nso].coef[0] = GLV(line,2);
  WSC[nso].coef[1] = GLV(line,3);
  WSC[nso].coef[2] = GLV(line,4);
  WSC[nso].coef[3] = GLV(line,5);
  fgetline(line,infu);
  WSC[nso].orderinv   = 3;
  WSC[nso].xoffinv    = GLV(line,1);
  WSC[nso].coefinv[0] = GLV(line,2);
  WSC[nso].coefinv[1] = GLV(line,3);
  WSC[nso].coefinv[2] = GLV(line,4);
  WSC[nso].coefinv[3] = GLV(line,5);
  fclose(infu);
}
return;
}


/* ----------------------------------------------------------------------
  Read in a general FITS image (nc,nr is expected size of image).
*/
void nsx_read_general_image( char ffile[], float image[], int nc, int nr )
{
/**/
int nnc,nnr,anynull,hdutype;
int status = 0;
/**/
float nullval = 0.;  /* don't check for null values in the image */
/**/
long nbuffer;
long firstpixel = 1;
/**/
fitsfile *fptr;
/**/
nbuffer = nc * nr;
fits_open_file( &fptr, ffile, READONLY, &status );
if (status != 0) { printf("***error: problem reading '%s'.\n",ffile); exit(1); }
fits_movabs_hdu( fptr, 1, &hdutype, &status );
nnc = cfua_inhead(fptr,"NAXIS1");
nnr = cfua_inhead(fptr,"NAXIS2");
if (verbose) printf("Reading FITS image '%s' (%d x %d).\n",ffile,nnc,nnr);
if ((nnc != nc)||(nnr != nr)) {
  printf("***error: image not expected size (nc=%d nr=%d nnc=%d nnr=%d).\n",nc,nr,nnc,nnr); exit(1);
}
fits_read_img( fptr, TFLOAT, firstpixel, nbuffer, &nullval, image, &anynull, &status ); cfua_error(status);
fits_close_file( fptr, &status ); cfua_error(status);
return;
}



/* ----------------------------------------------------------------------
 Smooth an array of (double precision) numbers using a boxcar.
 Input:   radius : radius of boxcar.
          num    : number of points.
 In/Out:  arr    : Array to be smoothed.
 Scratch: scr    : Scratch array.
*/
/*@@*/
void nsx_SmoothArray8( int radius, int num, double arr[], double scr[] )
{
/**/
int ii,jj;
double sum,wsum;
/**/
/* Copy. */
for (ii=0; ii<num; ++ii) { scr[ii] = arr[ii]; }
/* Smooth. */
for (ii=0; ii<num; ++ii) {
  sum =0.;
  wsum=0.;
  for (jj=(ii-radius); jj<=(ii+radius); ++jj) {
    if ((jj>=0)&&(jj<num)) {
      sum = sum + scr[jj];
      wsum= wsum+ 1.;
    }
  }
  if (wsum > 0.) arr[ii] = sum / wsum;
}
return;
}







/* ----------------------------------------------------------------------
 Smooth an image using a boxcar.    Ignore pixels < -9000.
 Input:   nc,nr  : number of columns and rows.
          img[]  : image to be smoothed (changed on output).
          ref[]  : scratch image of same size to be used as reference.
          colrad : radius of box (-colrad..0..colrad).  1,1 is a 3x3 box.
          rowrad : radius of box (-rowrad..0..rowrad).
 (Warning: the edge pixels of colrad and rowrad are ignored (for speed).)
*/
/*@@*/
void nsx_Smooth_Image( int nc, int nr, float img[], float ref[], int colrad, int rowrad )
{
/**/
int pixno,ii,jj,iii,jjj,ii1,ii2,jj1,jj2;
float sum,size;
/**/
/* Copy. */
for (ii=0; ii<(nc*nr); ++ii) { ref[ii] = img[ii]; }
/* Smooth. */
size = (1 + colrad + colrad) * (1 + rowrad + rowrad);
for (ii=colrad; ii<(nc-colrad); ++ii) {
  ii1=ii-colrad; ii2=ii+colrad;
  for (jj=rowrad; jj<(nr-rowrad); ++jj) {
    jj1=jj-rowrad; jj2=jj+rowrad;
    sum=0.;
    for (iii=ii1; iii<=ii2; ++iii) {
    for (jjj=jj1; jjj<=jj2; ++jjj) {
      pixno = iii + (jjj * nc);
      sum = sum + ref[pixno];
    }}
    pixno = ii + (jj * nc);
    img[pixno] = sum / size;
  }
}
return;
}


/* ----------------------------------------------------------------------
  Write out a FITS image.
*/
void nsx_write_general_image( char ffile[], float image[], int nc, int nr )
{
/**/
char wrd[100];
/**/
fitsfile *fptr;
/**/
int bitpix,hdutype;
int status = 0;
/**/
long nelements;
long naxis = 2;
long fpixel = 1;
long naxes[9];
/**/
/* Create a new FITS image. */
bitpix = -32;
naxes[0] = nc;
naxes[1] = nr;
strcpy(wrd,"!"); strcat(wrd,ffile);
if (verbose) printf("NOTE: Writing FITS image file '%s'.\n",ffile);
fits_create_file( &fptr, wrd, &status );                 cfua_error( status );
fits_create_img( fptr, bitpix, naxis, naxes, &status );  cfua_error( status );
fits_movabs_hdu( fptr, 1, &hdutype, &status );           cfua_error( status );
nelements = nc * nr;
fits_write_img( fptr, TFLOAT, fpixel, nelements, image, &status ); cfua_error(status);
fits_close_file( fptr, &status );                                  cfua_error(status);
return;
}



/* ----------------------------------------------------------------------
 Mash/median rows(1) or columns(2).
 Size of mash[] will be either 'nc'(mode=1) or 'nr'(mode=2).
 Mash will be between up1 and up2 (which are either row numbers(mode=1) 
 or column numbers(mode=2)).  These values are limited by bounds.
*/
void nsx_mash_median( int mode, float image[], int nc, int nr, double mash[], int up1, int up2 )
{
/**/
int pixno,ii,jj,p1,p2,narr;
/**/
float arr[9000];
/**/
if (mode == 1) {
  p1 = MIN((nr-1),MAX(0,(up1)));
  p2 = MIN((nr-1),MAX(0,(up2)));
  for (ii=0; ii<nc; ++ii) {
    narr=0;
    for (jj=p1; jj<=p2; ++jj) { 
      pixno=ii+(jj*nc); arr[narr]=image[pixno]; ++narr;
    }
    mash[ii]= cfind_median(narr,arr);
  }
}
if (mode == 2) {
  p1 = MIN((nc-1),MAX(0,(up1)));
  p2 = MIN((nc-1),MAX(0,(up2)));
  for (jj=0; jj<nr; ++jj) {
    narr=0;
    for (ii=p1; ii<=p2; ++ii) { 
      pixno=ii+(jj*nc); arr[narr]=image[pixno]; ++narr;
    }
    mash[jj]= cfind_median(narr,arr);
  }
}
return;
}


/* ----------------------------------------------------------------------
 Mash rows(1) or columns(2).
 Size of mash[] will be either 'nc'(mode=1) or 'nr'(mode=2).
 Mash will be between up1 and up2 (which are either row numbers(mode=1) 
 or column numbers(mode=2)).  These values are limited by bounds.
*/
void nsx_mash( int mode, float image[], int nc, int nr, double mash[], int up1, int up2 )
{
/**/
int pixno,ii,jj,p1,p2;
/**/
double sum,num;
/**/
if (mode == 1) {
  p1 = MIN((nr-1),MAX(0,(up1)));
  p2 = MIN((nr-1),MAX(0,(up2)));
  for (ii=0; ii<nc; ++ii) {
    sum=0.; num=0.;
    for (jj=p1; jj<=p2; ++jj) { pixno=ii+(jj*nc); sum=sum+image[pixno]; num=num+1.0; }
    mash[ii]= sum/num;
  }
}
if (mode == 2) {
  p1 = MIN((nc-1),MAX(0,(up1)));
  p2 = MIN((nc-1),MAX(0,(up2)));
  for (jj=0; jj<nr; ++jj) {
    sum=0.; num=0.;
    for (ii=p1; ii<=p2; ++ii) { pixno=ii+(jj*nc); sum=sum+image[pixno]; num=num+1.0; }
    mash[jj]= sum/num;
  }
}
return;
}


/* ----------------------------------------------------------------------
  Centroid 3 (with guess and radius).  
  Background is straight line fit to outer-most 6 pixels near radius limits.
*/
double nsx_centroid3( int nn, double xx[], double yy[], double guess, double radius, int niter, 
                      double *back, double *peak, float bckx[], float bcky[], int *bckn )
{
/**/
double coef[9],xoff,xxf[9],yyf[9],wwf[9];
double cent,sum,wsum,bb;
int iter,ii,ii1,ii2,kk,nnf;
/**/
for (ii=0; ii<4; ++ii) { wwf[ii]=1.; }
/* Iterations. */
cent=guess;  *peak=0.;
for (iter=0; iter<niter; ++iter) {
/* Range. */
  ii1 = cneari_bs( cent-radius, nn, xx );
  ii2 = cneari_bs( cent+radius, nn, xx );
  if (ii1 < 0) ii1=0;
  if (ii2 > (nn-1)) ii2=nn-1;

/* Straight line background. */
  xxf[0] = xx[ii1];
  xxf[1] = xx[ii1+1];
  xxf[2] = xx[ii2-1];
  xxf[3] = xx[ii2];
  yyf[0] = yy[ii1];
  yyf[1] = yy[ii1+1];
  yyf[2] = yy[ii2-1];
  yyf[3] = yy[ii2];
  nnf=4;
  if (GJ_polyfit(nnf,xxf,yyf,wwf,1,0,&xoff,coef) != 1) { printf("***error: fit failed:c3..\n"); exit(1); }

/* Peak. */
  for (ii=ii1; ii<=ii2; ++ii) { 
    if (yy[ii] > *peak) *peak = yy[ii];
  }

/* Centroid. */
  sum=0.; wsum=0.; kk=0;
  for (ii=ii1; ii<=ii2; ++ii) {
    bb  = cpolyval(2,coef,(xx[ii]-xoff));
    sum = sum + ( xx[ii] * (yy[ii] - bb) );
    wsum= wsum+ (yy[ii] - bb);
    bckx[kk] = xx[ii];
    bcky[kk] = bb;
    ++kk;
  }
  *bckn = kk;
  if (wsum > 0.) { cent = sum / wsum; } else { printf("***error: bad centroid.\n"); exit(1); }

}
*back= cpolyval(2,coef,(cent-xoff));
return(cent);
}



/* ----------------------------------------------------------------------
  Simple line flux.
*/
double nsx_lineflux( int nn, double xx[], double yy[], double cent, double radius )
{
/**/
double flux,low;
int ii;
/**/
low=9999999.;
for (ii=0; ii<nn; ++ii) {
  if ( (xx[ii] > (cent-radius))&&(xx[ii] < (cent+radius)) ) {
    if (yy[ii] < low) low=yy[ii]; 
  }
}
flux=0.;
for (ii=0; ii<nn; ++ii) {
  if ( (xx[ii] > (cent-radius))&&(xx[ii] < (cent+radius)) ) {
    flux = flux + (yy[ii] - low);
  }
}
return(flux);
}



/* ----------------------------------------------------------------------
  Compute approximate (1D) FWHM for an emission line.
  Returns -1.e+30 if problem.
*/
double nsx_fwhm_1D( int nn, double xx[], double yy[], double centroid, double radius, double *peak, int echo )
{
/**/
double fwhm,loyy,sum;
int ii,ii1,ii2;
/**/
/* Range. */
ii1 = cneari_bs( centroid-radius, nn, xx );
ii2 = cneari_bs( centroid+radius, nn, xx );
if (ii1 < 0) ii1=0;
if (ii2 > (nn-1)) ii2=nn-1;
/* Background. */
loyy=9.e+30; for (ii=ii1; ii<=ii2; ++ii) { if (yy[ii] < loyy) loyy=yy[ii]; }
sum=0.; *peak=0.;
for (ii=ii1; ii<=ii2; ++ii) {
  sum = sum + (yy[ii] - loyy);
  if (yy[ii] > *peak) *peak=yy[ii];
}
if (*peak > 0.) {
  fwhm = (0.939437 * sum) / *peak;
} else {
  if (echo) printf("===warning: bad FWHM (centroid=%f).\n",centroid);
  fwhm=-1.e+30;
}
return(fwhm);
}


/* ----------------------------------------------------------------------
  Centroid 2 (with guess and radius).  Background is lowest pixel within radius.
  Returns -1.e+30 if problem.
*/
double nsx_centroid2( int nn, double xx[], double yy[], double guess, double radius, int niter, int echo )
{
/**/
double cent,loyy,sum,wsum;
int iter,ii,ii1,ii2;
/**/
/* Iterations. */
cent = guess;
for (iter=0; iter<niter; ++iter) {
/* Range. */
  ii1 = cneari_bs( cent-radius, nn, xx );
  ii2 = cneari_bs( cent+radius, nn, xx );
  if (ii1 < 0) ii1=0;
  if (ii2 > (nn-1)) ii2=nn-1;
/* Background. */
  loyy=9.e+30; for (ii=ii1; ii<=ii2; ++ii) { if (yy[ii] < loyy) loyy=yy[ii]; }
/* Centroid. */
  sum=0.; wsum=0.;
  for (ii=ii1; ii<=ii2; ++ii) {
    sum = sum + ( xx[ii] * (yy[ii] - loyy) );
    wsum= wsum+ (yy[ii] - loyy);
  }
  if (wsum > 0.) { cent = sum / wsum; } else { 
    if (echo) printf("===warning: bad centroid (guess=%f).\n",guess); 
    cent=-1.e+30; 
  }
}
return(cent);
}



/* ----------------------------------------------------------------------
  Centroid.
*/
double nsx_centroid( int nn, double yy[], int uu1, int uu2 )
{
/**/
double cent,loyy,sum,wsum;
int ii,ii1,ii2;
/**/
ii1=uu1; ii2=uu2;
if (ii1 < 0) ii1=0;
if (ii2 > (nn-1)) ii2=nn-1;
loyy=9.e+30;
for (ii=ii1; ii<=ii2; ++ii) { if (yy[ii] < loyy) loyy=yy[ii]; }
sum=0.; wsum=0.;
for (ii=ii1; ii<=ii2; ++ii) { 
  sum = sum + ( (double)ii * (yy[ii] - loyy) );
  wsum= wsum+ (yy[ii] - loyy);
}
if (wsum > 0.) { cent = sum / wsum; } else { printf("***error: bad centroid.\n"); exit(1); }
return(cent);
}



/* ----------------------------------------------------------------------
  Find real value of lower or upper bound at a given column.
  mode=1 : lower bound.
  mode=2 : upper bound.
  The lower bound is offset for some orders to align profiles.
  NOTE: Shifting polynomials by 'yvoff' pixels inward (fudge).. -tab 14mar2017
*/
double nsx_find_real_image_row( int mode, int icol, int nso )
{
/**/
double xv,yv;
/**/
/*
double yvoff_lo[9] = { 0., 0., 0., 3., 2., 1., 1., 1., 0. };  /x OLD NUMBERS x/
double yvoff_up[9] = { 0., 0., 0., 3., 2., 1., 1., 1., 0. };  /x OLD NUMBERS x/
*/
double yvoff_lo[9] = { 0., 0., 0., 2., 1., 0., 0., 0., 0. };
double yvoff_up[9] = { 0., 0., 0., 5., 4., 3., 3., 3., 0. };
/**/
if (mode == 1) {
  xv = (double)icol - NCP[nso].xoffa;
  yv =cpolyval(NCP[nso].porda+1,NCP[nso].coefa,xv);
  yv = yv + yvoff_lo[nso];
} else {
  xv = (double)icol - NCP[nso].xoffb;
  yv =cpolyval(NCP[nso].pordb+1,NCP[nso].coefb,xv);
  yv = yv - yvoff_up[nso];
}
yv = yv + SlitOffset;
return(yv);
}



/* ----------------------------------------------------------------------
  Find image row given a column, nires spec order, and polynomials.
  mode=1 : lower bound.
  mode=2 : upper bound.
  The lower bound is offset for some orders to align profiles.
*/
int nsx_find_image_row( int mode, int icol, int nso )
{
int row;
row = cnint(( nsx_find_real_image_row(mode,icol,nso) ));
return(row);
}



/*----------------------------------------------------------------------
  Load TYC[]. 
  'TYC.bin' was created by some code in /home/tb/c/f/tst.c ... -tab August 2014
*/
void nsx_load_TYC( char nsxdir[], TYCtype TYC[], int *numTYC )
{
/**/
char tycfile[200];
FILE *binfu;
int nn;
/**/
sprintf(tycfile,"%scal/TYC.bin",nsxdir);
printf("Reading '%s'.\n",tycfile);
nn=0;
binfu = fopen_read(tycfile);
while ( fread((char *)&TYC[nn], 1, sizeof(TYCtype), binfu ) == sizeof(TYCtype) ) { ++nn; }
fclose(binfu);
*numTYC = nn;
printf("Read in %d Tycho entries.\n",*numTYC);
return;
}



/* ----------------------------------------------------------------------
 Set N2 for DAR_refwave (avoid having to calculate each time.
 Set the global variable N2_refwave  .
 This is good for 2km,P=600,T=7,f=8... (LAT=+/-30) from Allen ..
 (see nsx_difatmref)
*/
void nsx_set_N2_refwave()
{
/**/
double R,N2;
/**/
R = 1.e+4 / DAR_refwave;
N2= 64.328 + (29498.1 / (146.0-(R*R))) + (255.4 / (41.0-(R*R)));
N2= N2 * (600.0 * (1.0 + (1.049 - (0.0157*7.0)) * 600.0 * 1.e-6));
N2= N2 / (720.883 * (1.0 + (0.003661*7.0)));
N2_refwave = N2 / 1.e+6;
return;
}
 

/* ----------------------------------------------------------------------
  Differential atmospheric refraction.
  The following is good for 2km,P=600,T=7,f=8... (LAT=+/-30) from Allen ..
  wave = wavelength in Angstroms.
  ZA   = Apparent Zenith Angle in degrees.
  Returns 'DAR' differential atmospheric refraction in arcseconds, 
  positive values will be toward the blue relative to 'DAR_refwave'
  (which is defined at 12000 Angstroms for NIRES).
 
  NIRES: ROTPOSN is the rotator angle which points in the direction of
    increasing row number on the NIRES image given N=0deg and E=+90deg .
    This means that delta row value is (given the DAR and angles and ASPP)
      DROW = DAR * cos(theta) / ASPP,  where  theta = ROTPOSN - PARANG
*/
double nsx_difatmref( double wave, double ZA )
{
/**/
double DAR,R,N1;
/**/
R = 1.e+4 / wave;
N1= 64.328 + (29498.1 / (146.0-(R*R))) + (255.4 / (41.0-(R*R)));
N1= N1 * (600.0 * (1.0 + (1.049 - (0.0157*7.0)) * 600.0 * 1.e-6));
N1= N1 / (720.883 * (1.0 + (0.003661*7.0)));
N1= N1 / 1.e+6;
DAR = (2.06265e+5) * (N1 - N2_refwave) * tan(degtorad(ZA));
return(DAR);
}



/* ----------------------------------------------------------------------
  Find arcsec value for a given rowoff (and nso and column).
  'rowoff' is the real row position minus the edge position at given nso,icol..
  If IMG.el > 5. (degrees), then apply Differential Atmospheric Refraction..
  ( Returns arcsecond position of object with DAR.. i.e. 'AVP' computes the trace 
    of the object without any DAR.. the 'arcsec' value is given such that the
    actual centroid of the object is constant in arcsec space.) 
  is the return
*/
double nsx_AVP( AVPtype AVP[], int nso, int col, double rowoff, IMGtype IMG )
{
/**/
double xv,arcsec,zt,theta,wave;
int order = 2;
/**/
xv = rowoff - AVP[nso].xoff[col];
arcsec = cpolyval( order+1, AVP[nso].coef[col], xv );
if (IMG.el > 5.) {
  zt = 90. - IMG.el;
  theta = IMG.rotposn - IMG.parang;
  xv  = (double)col - WSC[nso].xoff;
  wave= cpolyval(WSC[nso].order+1,WSC[nso].coef,xv);
  arcsec = arcsec - (nsx_difatmref(wave,zt) * cos(degtorad(theta)));
}
return(arcsec);
}


/* ----------------------------------------------------------------------
  Find rowoff value for a given arcsec value (and nso and column).
  If IMG.el > 5. (degrees), then apply Differential Atmospheric Refraction..
  ( Returns 'rowoff' position of object as it appears on actual image data.
    So if a fixed arcsec is requested, that value will be adjusted by the DAR
    (before using AVP inverse formula) such as to follow the centroid of the 
    object along the slit. )
*/
double nsx_AVPinv( AVPtype AVP[], int nso, int col, double arcsec, IMGtype IMG )
{
/**/
double xv,rowoff,as,zt,wave,theta;
int order = 2;
/**/
as = arcsec;
if (IMG.el > 5.) {
  zt = 90. - IMG.el;
  theta = IMG.rotposn - IMG.parang;
  xv  = (double)col - WSC[nso].xoff;
  wave= cpolyval(WSC[nso].order+1,WSC[nso].coef,xv);
  as  = as + (nsx_difatmref(wave,zt) * cos(degtorad(theta)));
}
xv = as - AVP[nso].xoffinv[col];
rowoff = cpolyval( order+1, AVP[nso].coefinv[col], xv );
return(rowoff);
}


/*----------------------------------------------------------------------
  Load calibration info.   -tab 14may2013
*/
void nsx_load_cal( char nsxdir[], SLTtype SLT[], SOPtype SOP1[], SOPtype SOP2[], 
                   AVPtype AVP[], TYCtype TYC[], int *numTYC, char UseAVP[] )
{
/**/
int ii,nso,nc,nr;
/**/
char wrd[200];
char line[200];
/**/
FILE *infu;
/**/

/* Load slant calibration data. --- OBSOLETE */
nc=nc_Nominal; nr=200; 
for (nso=3; nso<=7; ++nso) {
  sprintf(wrd,"%scal/offimg_%d.fits",nsxdir,nso);
  SLT[nso].nc = nc;
  SLT[nso].image = (float *)calloc(((nc*nr)+1000),sizeof(float));
  nsx_read_general_image( wrd, SLT[nso].image, nc, nr );
}


/* Load SOP1 (Slant-Offset-Polynomials)..  --- NEW, replaces SLT */
sprintf(wrd,"%scal/SOP1.dat",nsxdir);  printf("Read '%s'.\n",wrd);
infu = fopen_read(wrd);
while (fgetline(line,infu)) {
  nso = GLV(line,1);
  ii  = GLV(line,2);
  SOP1[nso].xoff[ii]    = GLV(line,3);
  SOP1[nso].coef[ii][0] = GLV(line,4);
  SOP1[nso].coef[ii][1] = GLV(line,5);
  SOP1[nso].coef[ii][2] = GLV(line,6);
}
fclose(infu);

/* Load SOP2 (Slant-Offset-Polynomials)..  --- correction to SOP1[] using skylines.. */
sprintf(wrd,"%scal/SOP2.dat",nsxdir);  printf("Read '%s'.\n",wrd);
infu = fopen_read(wrd);
while (fgetline(line,infu)) {
  nso = GLV(line,1);
  ii  = GLV(line,2);
  SOP2[nso].xoff[ii]    = GLV(line,3);
  SOP2[nso].coef[ii][0] = GLV(line,4);
  SOP2[nso].coef[ii][1] = GLV(line,5);
  SOP2[nso].coef[ii][2] = GLV(line,6);
}
fclose(infu);


/* Load AVP (arcsec vs. pixel) polnomials.. */
sprintf(wrd,"%scal/AVP.%s.dat",nsxdir,UseAVP);
printf("Read '%s'.\n",wrd);
infu = fopen_read(wrd);
while (fgetline(line,infu)) {
  nso = GLV(line,1);
  ii  = GLV(line,2);
  AVP[nso].xoff[ii]    = GLV(line,3);
  AVP[nso].coef[ii][0] = GLV(line,4);
  AVP[nso].coef[ii][1] = GLV(line,5);
  AVP[nso].coef[ii][2] = GLV(line,6);
}
fclose(infu);
sprintf(wrd,"%scal/AVPinv.%s.dat",nsxdir,UseAVP);
printf("Read '%s'.\n",wrd);
infu = fopen_read(wrd);
while (fgetline(line,infu)) {
  nso = GLV(line,1);
  ii  = GLV(line,2);
  AVP[nso].xoffinv[ii]    = GLV(line,3);
  AVP[nso].coefinv[ii][0] = GLV(line,4);
  AVP[nso].coefinv[ii][1] = GLV(line,5);
  AVP[nso].coefinv[ii][2] = GLV(line,6);
}
fclose(infu);



/* Load Tycho catalog.  */
nsx_load_TYC( nsxdir, TYC, numTYC );


/* Load wavelength scale. */
nsx_load_WSC( nsxdir );


/* Load polynomials into structure. */
/* These polynomials determine the edges of the slit for each order. */
sprintf(wrd,"%scal/nsx_edge.dat",nsxdir);  
if (VERB) printf("Read '%s'.\n",wrd);
infu = fopen_read(wrd);
for (nso=3; nso<=7; ++nso) {
  fgetline(line,infu);
  NCP[nso].porda = 3;
  NCP[nso].xoffa = GLV(line,2); 
  NCP[nso].coefa[0]=GLV(line,3); NCP[nso].coefa[1]=GLV(line,4); 
  NCP[nso].coefa[2]=GLV(line,5); NCP[nso].coefa[3]=GLV(line,6);
  fgetline(line,infu);
  NCP[nso].pordb = 3;
  NCP[nso].xoffb = GLV(line,2); 
  NCP[nso].coefb[0]=GLV(line,3); NCP[nso].coefb[1]=GLV(line,4); 
  NCP[nso].coefb[2]=GLV(line,5); NCP[nso].coefb[3]=GLV(line,6);
}
fclose(infu);

return;
}




/*----------------------------------------------------------------------
Command line syntax.
*/
void nsx_syntax()
{
printf("Syntax: nsx (raw NIRES FITS files)  [ NIRES B file ] [ sp=as,as ] [ bk=as,as ] [ xsp=pix,pix ] [ xbk=pix,pix ] \n");
printf("\n");
printf("  The optional 'B file' is another raw FITS file where the object has\n");
printf("    been moved to another position on the slit.\n");
printf("  If 1st argument ends with '.ls' then program assumes this is a file with\n");
printf("    a list of FITS files.\n");
printf("  Create a 'nsx.tbl' exposure listing using 'nsx file.ls list=nsx.tbl', where\n");
printf("    'file.ls' is a listing of all raw NIRES FITS files.\n");
printf("\n");
printf("  You must set NSXDIR (code dir) and NSXOUT (working dir) environment variables.\n");
printf("\n");
printf("Options:\n");
printf("  sp=    : Specify region to spectrally extract in relative arcseconds.\n");
printf("  bk=    : Specify region for background in relative arcseconds (multiple bk= may be used).\n");
printf("  xsp=   : Specify region to spectrally extract (pixels on reddest order#3).\n");
printf("  xbk=   : Specify region for background (pixels on reddest order #3) (multiple xbk= may be used).\n");
printf("  list=  : Specify a file to which to write a 'table' listing of exposures.\n");
printf("  -list  : Table listing to standard output.\n");
printf("  -autox : Automatically select object extraction range and extract spectrum.\n");
printf("  sfx=   : Suffix to label output files.\n");
printf("  log=   : Name of log file (default will be root of first FITS file with '.log' extension).\n");
printf("-NoFlat     : Do not correct for pixel-to-pixel flat field variations.\n");
printf("-NoClean    : Do not correct for cosmic rays (or other artifacts).\n");
printf("-NoHotClean : Do not correct for Hot Pixels.\n");
printf("\nRecalibration:\n");
printf("-calavp  : Calibrate Arcsecs Vs. Pixels (build data file using bright stars).\n");
printf("-newavp  : Using built data file, create a new AVP calibration set.\n");
printf("   avp=  : Use a specific AVP root code (e.g. 2018-01-20).\n");
/*
printf("  -autob : Automatically select a B file (if a 'nsx.tbl' file exists).\n");
printf("  -noplot: Do not create PostScript plots (use if nsxplot not created).\n");
printf("  -noback: No background subtraction.\n");
printf("  bk=    : Specify region for background in arcseconds (any order).\n");
printf("           (bk= and abk= options may be repeated)\n");
printf("  -rbi   : Write out re-binned image with variance and wavelengths.\n");
printf("  ac=    : Root (e.g. 'nires0045') of a Vega-type star for atmospheric correction.\n");
printf("darklist=: Create hot pixel map using these (short) darks.\n");
*/
printf("[ NSX Version: %s ]\n",NSXVERSION);
printf("\n");

return;
}


/* ----------------------------------------------------------------------
 Median filter.   -tab 03apr2014
*/
void nsx_median_filter( int bin, int nn, double arr[], double mfarr[] )
{
/**/ 
int hbin,ii,ii1,ii2,jj,tnn;
/**/ 
const int mtarr=29;
double tarr[mtarr];
double median;
/**/ 
if (bin > mtarr-2) { printf("***error: bin too large.\n"); exit(1); }
hbin=bin/2;
for (ii=0; ii<nn; ++ii) {
  ii1 = ii-hbin;
  ii2 = ii+hbin;
  if (ii1 < 0   ) ii1=0;
  if (ii2 > nn-1) ii2=nn-1;
  tnn=0;
  for (jj=ii1; jj<=ii2; ++jj) { tarr[jj-ii1] = arr[jj]; ++tnn; }
  median = cfind_median8(tnn,tarr); 
  mfarr[ii] = median;
}
return;
}



/* ----------------------------------------------------------------------
  Find centroid from dif[] array.  mode=0 (absorption)  mode=1 (emission)
*/
double nsx_dif_cent( int nn, double xx[], double yy[], double guess, int mode )
{
/**/
double sw,wgt,sum,wsum,cent;
int ii0,ii1,ii2,ii;
/**/

/* Emission or absorption? */
if (mode) { sw=1.; } else { sw=-1.; }

/* 1st iteration. */
ii0 = cneari_bs( guess, nn, xx );
ii1 = ii0 - 3;
ii2 = ii0 + 3;
sum=0.; wsum=0.;
for (ii=ii1; ii<=ii2; ++ii) {
  wgt = sw * yy[ii];
  if (wgt > 0.) {
    sum = sum + (xx[ii] * wgt);
    wsum= wsum+ wgt;
  }
}
if (wsum < 0.00001) { printf("***error: nsx_dif_cent: wsum too small.\n"); exit(1); }
cent = sum / wsum;

/* 2nd iteration. */
ii0 = cneari_bs( cent, nn, xx );
ii1 = ii0 - 4;
ii2 = ii0 + 4;
sum=0.; wsum=0.;
for (ii=ii1; ii<=ii2; ++ii) {
  wgt = sw * yy[ii];
  if (wgt > 0.) {
    sum = sum + (xx[ii] * wgt);
    wsum= wsum+ wgt;
  }
}
if (wsum < 0.00001) { printf("***error: nsx_dif_cent: wsum too small.\n"); exit(1); }
cent = sum / wsum;

return(cent);
}


/* ----------------------------------------------------------------------
 Find background region from a profile (also find test object region).
*/
void nsx_background_region( int nso, int numpro, double npro[9][300], short nbck[9][300], short nobj[9][300] )
{
/**/
int narr,ii,kk;
/**/
float p95,p80,median,arr[300];
/**/
/* Clear, set up, and sort. */
for (ii=0; ii<300; ++ii) { nbck[nso][ii]=0; nobj[nso][ii]=0; }
narr=0;
for (ii=0; ii<numpro; ++ii) { arr[narr]=npro[nso][ii]; ++narr; }
median=cfind_median(narr,arr); 
kk=0.80*(narr-1); p80=arr[kk];
kk=0.95*(narr-1); p95=arr[kk];
for (ii=0; ii<numpro; ++ii) {
  if (npro[nso][ii] < p80) { nbck[nso][ii] = 1; }
  if (npro[nso][ii] > p95) { nobj[nso][ii] = 1; }
}
return;
}


/* ----------------------------------------------------------------------
  Find minimum width for this echelle order.
  Changed 'width = 1 + upper - lower' to 'width = upper - lower' -tab 30nov2017
*/
int nsx_minwidth( int ecol, int nso )
{
/**/
int ii,ilo;
double lower,upper,width,lo;
/**/
lo=999.;
for (ii=20; ii<ecol-20; ii=ii+50) {
  lower = nsx_find_real_image_row( 1, ii, nso );
  upper = nsx_find_real_image_row( 2, ii, nso );
  width = upper - lower;
  if (width < lo) { lo=width; }
}
if (lo <   1.) { printf("***error:a: lo<1..\n"); exit(1); }
if (lo > 900.) { printf("***error:b: lo>900..\n"); exit(1); }
ilo = cnint(lo);
return(ilo);
}


/* ----------------------------------------------------------------------
  Median within sub-image.  -tab 09jan2018 
*/
double nsx_submedian( float image[], int nc, int nr, int irad, int jrad, int ii0, int jj0 )
{
/**/
int pixno,narr,ii1,ii2,jj1,jj2,ii,jj;
float arr[(nc*nr)];
double median;
/**/

/* Boundaries. */
ii1=ii0-irad; ii2=ii0+irad;
jj1=jj0-jrad; jj2=jj0+jrad;
if (ii1 <   0  ) ii1=0;
if (ii2 > nc-1 ) ii2=nc-1;
if (jj1 <   0  ) jj1=0;
if (jj2 > nr-1 ) jj2=nr-1;

/* Median. */
narr=0;
for (ii=ii1; ii<=ii2; ++ii) {
  for (jj=jj1; jj<=jj2; ++jj) {
    if ((ii != ii0)||(jj != jj0)) {
      pixno = ii + (jj1 * nc); 
      arr[narr] = image[pixno]; 
      ++narr;
    }
  }
}
if (narr < 1) { median=0.; narr=1; } else { median = cfind_median(narr,arr); }

return(median);
}




/* ----------------------------------------------------------------------
 Compute rms sigmas deviation from median near a pixel.  -tab 09jan2018 
*/
double nsx_rmssigdev( float image[], int nc, int nr, int irad, int jrad, int ii0, int jj0 )
{
/**/
int pixno,narr,kk,ii1,ii2,jj1,jj2,ii,jj;
float median,rms,arr[(nc*nr)];
double sigs;
/**/

/* Boundaries. */
ii1=ii0-irad; ii2=ii0+irad;
jj1=jj0-jrad; jj2=jj0+jrad;
if (ii1 <   0  ) ii1=0;
if (ii2 > nc-1 ) ii2=nc-1;
if (jj1 <   0  ) jj1=0;
if (jj2 > nr-1 ) jj2=nr-1;

/* Exclude center pixel. */
narr=0;
for (ii=ii1; ii<=ii2; ++ii) {
  for (jj=jj1; jj<=jj2; ++jj) {
    if ((ii != ii0)||(jj != jj0)) {
      pixno = ii + (jj * nc); 
      arr[narr] = image[pixno]; 
      ++narr;
    }
  }
}
if (narr < 1) { median=0.; narr=1; } else { median = cfind_median(narr,arr); }

/* RMS from median. */
rms=0.;
for (kk=0; kk<narr; ++kk) {
  rms = rms + ((arr[kk] - median) * (arr[kk] - median));
}
rms = sqrt(( rms / (double)narr ));

/* Sigmas. */
pixno = ii0 + (jj0 * nc);
sigs = (image[pixno] - median) / rms;

return(sigs);
}



/* ----------------------------------------------------------------------
 Stats on divided flats. 
*/
double nsx_flat_stats( float image1[], float image2[], float image3[], 
                     int nc, int nr, int sc, int ec, int sr, int er )
{
/**/
int ii,jj,pixno;
/**/
double rms,sum1,sum2,sum3,num,mean1,mean2,mean3,eperdn;
/**/

/* Check and echo. */
if ((sc < 0)||(sc > nc-1)||(ec < 0)||(ec > nc-1)) { printf("***error: columns out of range.\n"); exit(1); }
if ((sr < 0)||(sr > nr-1)||(er < 0)||(er > nr-1)) { printf("***error:    rows out of range.\n"); exit(1); }

/* Means. */
sum1=0.; sum2=0.; sum3=0.; num=0.;
for (ii=sc; ii<=ec; ++ii) {
for (jj=sr; jj<=er; ++jj) {
  pixno=ii+(jj*nc);
  sum1=sum1+image1[pixno];
  sum2=sum2+image2[pixno];
  sum3=sum3+image3[pixno];
  num =num + 1.0;
}}
if (num < 1.) { printf("***error: tfs: too few pixels.\n"); exit(1); }
mean1=sum1/num;
mean2=sum2/num;
mean3=sum3/num;

/* RMS of 3. */
sum3=0.;
for (ii=sc; ii<=ec; ++ii) {
for (jj=sr; jj<=er; ++jj) {
  pixno=ii+(jj*nc);
  sum3 = sum3 + ( (image3[pixno] - mean3) * (image3[pixno] - mean3) );
}}
rms = sqrt( sum3 / num );

eperdn = 2.0 / ( mean1 * rms * rms );

printf(" %12.4f %12.4f %12.8f %12.8f %12.6f \n",mean1,mean2,mean3,rms,eperdn);

return(eperdn);
}



/*----------------------------------------------------------------------
 Dark correction with ad hoc scaling.          -tab 11apr2014
 ( Generally do not use, since A-B does good job of getting rid of dark... )
*/
void nsx_correct_dark( float image[], int nc, int nr, float dark[], float scrimg[] )
{
/**/
int ii,jj,kk,pixno;
float bck,bck_dark,median;
/**/
/* Find background in upper left corner. */
kk=0;
for (ii=12; ii<91; ++ii) {
for (jj=925; jj<986; ++jj) {
  pixno = ii + (jj*nc);
  scrimg[kk] = image[pixno];
  ++kk;
}}
median = cfind_median(kk,scrimg);
ii = (float)kk * 0.10;
bck = scrimg[ii];
/* Same statistic in dark image. */
kk=0;
for (ii=12; ii<91; ++ii) {
for (jj=925; jj<986; ++jj) {
  pixno = ii + (jj*nc);
  scrimg[kk] = dark[pixno];
  ++kk;
}}
median = cfind_median(kk,scrimg);
ii = (float)kk * 0.10;
bck_dark = scrimg[ii];
printf("NOTE: Correct dark, scale by (img/dark): %0.3f / %0.3f = %0.3f .\n",bck,bck_dark,bck/bck_dark);
/* Scale dark and subtract off image. */
for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) {
  pixno = ii + (jj*nc);
  image[pixno] = image[pixno] - (dark[pixno] * (bck / bck_dark));
}}
return;
}



/* ----------------------------------------------------------------------
  Find object and background windows for an A - B profile.  -tab 07feb2018 
  Also works when B exposure does not exist.  -tab 08feb2018
  Returns '1' if there is a problem.
*/
int nsx_auto_window( IMGtype IA, IMGtype IB, char IABroot[], SPXtype SPX[], SPXtype SPXB[], int noback, char nsxout[] )
{
/**/
int iter,kk,good,ii,narr,nn,jj,ok;
int count,again,nbk,flag,kk1,kk2,ii1,ii2,hikk1,hikk2;
int nso,kk1x,kk2x,bokk1,bokk2,bnkk1,bnkk2;
int neg[MAXPRO],xobj[MAXPRO],obj[MAXPRO],msk[MAXPRO],bck[MAXPRO],bckadj[MAXPRO];
/**/
double xx[MAXPRO],yy[MAXPRO],median,arr[MAXPRO];
double abk1[MAXabk],abk2[MAXabk];
double radius,pxn,px0,guess,ratio,sca,peak,bopeak,bosum,bosigs,bnsum,bnsigs_limit,bnsigs;
double midpt,sum2,sigs,sum,losum,hisum,dif,rms,num;
/**/
FILE *drawfu;
char wrd[100];


/* Echo */
if (IB.X) {
  printf("Find object and background automatically from A - B profile..\n");
  sprintf(wrd,"%s%s-pro.draw",nsxout,IABroot); drawfu = fopen_write(wrd);
} else {
  printf("Find object and background automatically from profile..\n");
  sprintf(wrd,"%s%s-pro.draw",nsxout,IA.root); drawfu = fopen_write(wrd);
}

/* Median of profile. */
narr=0; nn=SPX[0].pro_apn;
for (ii=0; ii<nn; ++ii) { 
  xx[ii]=SPX[0].pro_apx[ii]; 
  yy[ii]=SPX[0].pro_apymed[ii] - SPXB[0].pro_apymed[ii];
  arr[narr]=yy[ii]; 
  ++narr; 
}
median = cfind_median8(narr,arr);
printf("initial median=%f  narr=%d  \n",median,narr);


fprintf(drawfu,"%f %f\n",-9.,median);
fprintf(drawfu,"%f %f\n",300.,median);
fprintf(drawfu,"draw\n");


/* Clear masking. */
for (ii=0; ii<nn; ++ii) { neg[ii]=0; xobj[ii]=0; obj[ii]=0; bck[ii]=0; msk[ii]=0; }


/* Find brightest object. */
hisum=0.; bokk1=0; bokk2=0;
for (ii=2; ii<nn-2; ++ii) { 
  kk1=ii-4; if (kk1 <   0 ) kk1=0;
  kk2=ii+4; if (kk2 > nn-1) kk2=nn-1;
  sum=0.;
  for (jj=kk1; jj<=kk2; ++jj) { sum = sum + yy[jj]; }
  if (sum > hisum) { hisum=sum; bokk1=kk1; bokk2=kk2; }
}
fprintf(drawfu,"sci 7; sls 4\n");
fprintf(drawfu,"%f %f\n",xx[bokk1],0.);
fprintf(drawfu,"%f %f\n",xx[bokk1],999000.);
fprintf(drawfu,"draw\n");
fprintf(drawfu,"%f %f\n",xx[bokk2],0.);
fprintf(drawfu,"%f %f\n",xx[bokk2],999000.);
fprintf(drawfu,"draw\n");

/* Expand main object. */
ii2 = bokk1-1;  if (ii2 < 0) ii2=0;
ii1 = bokk1-8; if (ii1 < 0) ii1=0;
flag= 1; 
for (ii=ii2; ii>=ii1; --ii) {
  if ((flag)&&(yy[ii] > median)) { bokk1=ii; } else { flag=0; }
}
ii1 = bokk2+1; if (ii1 > nn-1) ii1=nn-1;
ii2 = bokk2+8; if (ii2 > nn-1) ii2=nn-1;
flag= 1; 
for (ii=ii1; ii<=ii2; ++ii) {
  if ((flag)&&(yy[ii] > median)) { bokk2=ii; } else { flag=0; }
}
fprintf(drawfu,"sci 3; sls 1\n");
fprintf(drawfu,"%f %f\n",xx[bokk1],0.);
fprintf(drawfu,"%f %f\n",xx[bokk1],999000.);
fprintf(drawfu,"draw\n");
fprintf(drawfu,"%f %f\n",xx[bokk2],0.);
fprintf(drawfu,"%f %f\n",xx[bokk2],999000.);
fprintf(drawfu,"draw\n");

/* Mask main object pixels plus wings. */
kk1x = bokk1 - 4; if (kk1x < 0   ) kk1x=0;
kk2x = bokk2 + 4; if (kk2x > nn-1) kk2x=nn-1;
for (ii=0; ii<nn; ++ii) { if ((ii >= kk1x)&&(ii <= kk2x)) { obj[ii]=1; msk[ii]=1; } }


/* Find brightest negative object. */
bnkk1=0; bnkk2=0;
if (IB.X) {
  losum=0.;
  for (ii=2; ii<nn-2; ++ii) {
    kk1=ii-4; if (kk1 <   0 ) kk1=0;
    kk2=ii+4; if (kk2 > nn-1) kk2=nn-1;
    sum=0.;
    for (jj=kk1; jj<=kk2; ++jj) { sum = sum + yy[jj]; }
    if (sum < losum) { losum=sum; bnkk1=kk1; bnkk2=kk2; }
  }
  fprintf(drawfu,"sci 7; sls 2\n");
  fprintf(drawfu,"%f %f\n",xx[bnkk1],0.);
  fprintf(drawfu,"%f %f\n",xx[bnkk1],999000.);
  fprintf(drawfu,"draw\n");
  fprintf(drawfu,"%f %f\n",xx[bnkk2],0.);
  fprintf(drawfu,"%f %f\n",xx[bnkk2],999000.);
  fprintf(drawfu,"draw\n");
  
/* Expand main negative object. */
  ii2 = bnkk1-1;  if (ii2 < 0) ii2=0;
  ii1 = bnkk1-8; if (ii1 < 0) ii1=0;
  flag= 1;
  for (ii=ii2; ii>=ii1; --ii) {
    if ((flag)&&(yy[ii] < median)) { bnkk1=ii; } else { flag=0; }
  }
  ii1 = bnkk2+1; if (ii1 > nn-1) ii1=nn-1;
  ii2 = bnkk2+8; if (ii2 > nn-1) ii2=nn-1;
  flag= 1;
  for (ii=ii1; ii<=ii2; ++ii) {
    if ((flag)&&(yy[ii] < median)) { bnkk2=ii; } else { flag=0; }
  }
  fprintf(drawfu,"sci 3; sls 2\n");
  fprintf(drawfu,"%f %f\n",xx[bnkk1],0.);
  fprintf(drawfu,"%f %f\n",xx[bnkk1],999000.);
  fprintf(drawfu,"draw\n");
  fprintf(drawfu,"%f %f\n",xx[bnkk2],0.);
  fprintf(drawfu,"%f %f\n",xx[bnkk2],999000.);
  fprintf(drawfu,"draw\n");
  
  fprintf(drawfu,"sci 8; sls 4\n");
  fprintf(drawfu,"%f %f\n",-99.,median);
  fprintf(drawfu,"%f %f\n",300.,median);
  fprintf(drawfu,"draw\n");

/* Mask negative object pixels plus wings. */
  kk1x = bnkk1 - 4; if (kk1x < 0   ) kk1x=0;
  kk2x = bnkk2 + 4; if (kk2x > nn-1) kk2x=nn-1;
  for (ii=0; ii<nn; ++ii) { if ((ii >= kk1x)&&(ii <= kk2x)) { neg[ii]=1; msk[ii]=1; } }

}


/* Median and RMS of background avoiding main and negative objects. */
narr=0;
for (ii=0; ii<nn; ++ii) {
  if (msk[ii] == 0) { arr[narr]=yy[ii]; ++narr; } 
}
median = cfind_median8(narr,arr);
rms=0.; sum=0.; num=0.;
for (ii=0; ii<nn; ++ii) { if (msk[ii] == 0) {
  sum = sum + ((yy[ii] - median) * (yy[ii] - median));
  num = num + 1.;
}}
if (num > 0.) { rms = sqrt(( sum / num )); }
printf("Non-main median=%f  num=%f  rms=%f  \n",median,num,rms);

/* Flux and sigs of main object.. */
bosum=0.; sum2=0.; bopeak=0.;
for (ii=bokk1; ii<=bokk2; ++ii) { 
  bosum = bosum + (yy[ii]-median); 
  sum2  = sum2  + (rms*rms); 
  if (yy[ii] > bopeak) bopeak=yy[ii];
}
bosigs=0.; if (sum2 > 0.) { bosigs = bosum / sqrt(sum2); }
printf("Main: %0.2f  bosigs=%0.2f \n", bosum,bosigs);

/* Flux and sigs of negative object.. */
if (IB.X) {
  bnsum=0.; sum2=0.;
  for (ii=bnkk1; ii<=bnkk2; ++ii) { bnsum=bnsum+(yy[ii]-median); sum2=sum2+(rms*rms); }
  bnsigs=0.; if (sum2 > 0.) { bnsigs = ABS((bnsum)) / sqrt(sum2); }
  bnsigs_limit = bosigs / 10.;    /* Limit to determine if negative object is significant. */
  if (bnsigs_limit < 5.) bnsigs_limit = 5.;
  printf("Negative: %0.2f  bnsigs=%0.2f [limit=%0.2f]\n",bnsum,bnsigs,bnsigs_limit);
/* If negative object not-significant, erase negative object. */
  if (bnsigs < bnsigs_limit) {
    printf("Negative object not-significant, erase mask.\n");
    for (ii=0; ii<nn; ++ii) { if ((neg[ii] == 1)&&(obj[ii] == 0)) { neg[ii]=0; msk[ii]=0; } }
/* New Median and RMS of background avoiding main object only. */
    narr=0;
    for (ii=0; ii<nn; ++ii) { if (msk[ii] == 0) { arr[narr]=yy[ii]; ++narr; } }
    median = cfind_median8(narr,arr);
    rms=0.; sum=0.; num=0.;
    for (ii=0; ii<nn; ++ii) { if (msk[ii] == 0) {
      sum = sum + ((yy[ii] - median) * (yy[ii] - median));
      num = num + 1.;
    }}
    if (num > 0.) { rms = sqrt(( sum / num )); }
    printf("NEW Non-main-object median=%f  num=%f  rms=%f  \n",median,num,rms);
/* New Flux and sigs of main object.. */
    bosum=0.; sum2=0.;
    for (ii=bokk1; ii<=bokk2; ++ii) { bosum=bosum+(yy[ii]-median); sum2=sum2+(rms*rms); }
    bosigs=0.; if (sum2 > 0.) { bosigs = bosum / sqrt(sum2); }
    printf("NEW Main: %0.2f  bosigs=%0.2f  \n",bosum,bosigs);
  }
}


fprintf(drawfu,"sci 2; sls 1\n");
fprintf(drawfu,"%f %f\n",-99.,median);
fprintf(drawfu,"%f %f\n",300.,median);
fprintf(drawfu,"draw\n");
fprintf(drawfu,"sci 2; sls 4\n");
fprintf(drawfu,"%f %f\n",-99.,median+rms);
fprintf(drawfu,"%f %f\n",300.,median+rms);
fprintf(drawfu,"draw\n");
fprintf(drawfu,"%f %f\n",-99.,median-rms);
fprintf(drawfu,"%f %f\n",300.,median-rms);
fprintf(drawfu,"draw\n");


/* Set main object limits. */
SPX[0].nsp     = 1;
SPX[0].asp1[0] = xx[bokk1];
SPX[0].asp2[0] = xx[bokk2];
if (IA.exptime > 0.) { SPX[0].pflx[0] = bosum / IA.exptime; }
SPX[0].peak[0] = bopeak;
SPX[0].sigs[0] = bosigs;


/* Centroid main object in each echelle order median profile. */
/* Note that only the SPX[0].pro_cen value is really used anywhere (in TraceAVP). */
radius = ( SPX[0].asp2[0] - SPX[0].asp1[0] ) / 2.;
guess  = ( SPX[0].asp2[0] + SPX[0].asp1[0] ) / 2.;
SPX[0].pro_cen = nsx_centroid2( SPX[0].pro_apn, SPX[0].pro_apx, SPX[0].pro_apymed, guess, radius, 3, 1 );
px0 = SPX[0].pro_cen / ARCSEC_PER_PIXEL;
printf("pro_cen(nso=0): %8.4f [%8.4f px]\n",SPX[0].pro_cen, px0 );
for (nso=3; nso<=7; ++nso) {
  SPX[nso].pro_cen = nsx_centroid2( SPX[nso].pro_apn, SPX[nso].pro_apx, SPX[nso].pro_apymed, guess, radius, 3, 1 );
  pxn = SPX[nso].pro_cen / ARCSEC_PER_PIXEL;
  printf("pro_cen(nso=%d): %8.4f [%8.4f px] (%8.4f)\n",nso,SPX[nso].pro_cen, pxn, pxn - px0 );
}




/* Find other objects. */
again=1; count=1;
while (again) {

/* Find next brightest object. */
  hisum=0.; hikk1=0; hikk2=0;
  for (ii=2; ii<nn-2; ++ii) {
    kk1=ii-4; if (kk1 <   0 ) kk1=0;
    kk2=ii+4; if (kk2 > nn-1) kk2=nn-1;
    sum=0.; ok=1;
    for (jj=kk1; jj<=kk2; ++jj) {
      if (obj[jj]||msk[jj]) ok=0;
      sum = sum + yy[jj];
    }
    if (ok) {
      if (sum > hisum) { hisum=sum; hikk1=kk1; hikk2=kk2; }
    }
  }

/* Is this object significant? */
  sum=0.; sum2=0.; peak=0.;
  for (ii=hikk1; ii<=hikk2; ++ii) {
    sum = sum + (yy[ii] - median);
    sum2= sum2+ (rms * rms);
    if (yy[ii] > peak) peak=yy[ii];
  }
  sigs=0.;
  if (sum2 > 0.) { sigs = sum / sqrt(sum2); }
  printf("next object: sum=%f  sigs=%f  (%f %f)\n",sum,sigs,xx[hikk1],xx[hikk2]);

/* Mask if significant. */
  if (sigs > 4.) {
    again = 1;
    for (ii=0; ii<nn; ++ii) { if ((ii >= hikk1)&&(ii <= hikk2)) { msk[ii]=1; } }
/* Does this look like a real object? */
    good=1; kk=(hikk1+hikk2)/2; midpt = 0.;
    for (ii=kk-2; ii<=kk+2; ++ii) { if (yy[ii] > midpt) midpt=yy[ii]; }
    if ((midpt < yy[hikk1])||(midpt < yy[hikk2])) { good=0; }
    if (good) {
      SPX[0].asp1[(SPX[0].nsp)] = xx[hikk1];
      SPX[0].asp2[(SPX[0].nsp)] = xx[hikk2];
      if (IA.exptime > 0.) { SPX[0].pflx[(SPX[0].nsp)] = sum / IA.exptime; }
      SPX[0].sigs[(SPX[0].nsp)] = sigs;
      SPX[0].peak[(SPX[0].nsp)] = peak;
      SPX[0].nsp = SPX[0].nsp + 1;
      for (ii=0; ii<nn; ++ii) { if ((ii >= hikk1)&&(ii <= hikk2)) { xobj[ii]=1; } }
fprintf(drawfu,"sls 4; sci %d\n",3+count);
fprintf(drawfu,"%f %f\n",xx[hikk1],0.);
fprintf(drawfu,"%f %f\n",xx[hikk1],999000.);
fprintf(drawfu,"draw\n");
fprintf(drawfu,"%f %f\n",xx[hikk2],0.);
fprintf(drawfu,"%f %f\n",xx[hikk2],999000.);
fprintf(drawfu,"draw\n");
printf(" a sig obj = %f %f\n",xx[hikk1],xx[hikk2]);
    }
  } else {
    again = 0;
  }
  ++count;

}


/* Find background pixels based on closeness to median. */
iter=1; count=0; sca=1.0; ratio=0.0001;
while ((iter < 9)&&(ratio < 0.4)) {
  for (ii=0; ii<nn; ++ii) { if ((bck[ii] == 0)&&(obj[ii] == 0)&&(neg[ii] == 0)) {
    dif = ABS(( yy[ii] - median ));
    if (dif < sca*rms) { bck[ii]=1; ++count; }
  }}
  ratio = (double)count / (double)nn;
  printf("sca=%6.3f  iter=%d  count=%d  ratio=%f \n",sca,iter,count,ratio);
  sca = 1.1 * sca;
  ++iter;
}


fprintf(drawfu,"sci 5 ; sym 4; sch 2.0\n");
for (ii=0; ii<nn; ++ii) { if (bck[ii]) {
  fprintf(drawfu,"%f %f\n",xx[ii],yy[ii]);
}}
fprintf(drawfu,"plot\n");


/* Add in surrounding pixels.  */
count=0;
for (ii=0; ii<nn; ++ii) { bckadj[ii]=0; }
for (ii=0; ii<nn; ++ii) { 
  if ((obj[ii] == 0)&&(neg[ii] == 0)&&(bck[ii] == 0)) {
    dif = ABS(( yy[ii] - median ));
    if (dif < 2.*rms) {
      jj=ii-1; if (jj < 0) jj=0;
      if (bck[jj]) bckadj[ii] = 1;
      jj=ii+1; if (jj > nn-1) jj=nn-1;
      if (bck[jj]) bckadj[ii] = 1;
      jj=ii-2; if (jj < 0) jj=0;
      if (bck[jj]) bckadj[ii] = 1;
      jj=ii+2; if (jj > nn-1) jj=nn-1;
      if (bck[jj]) bckadj[ii] = 1;
    }
  }
}
for (ii=0; ii<nn; ++ii) { 
  if (bckadj[ii]) { ++count; bck[ii]=1; }
}
printf("added %d adjacent background pixels.\n",count);


/* Create background regions. */
flag=0; nbk=0; kk1=0; kk2=0;
for (ii=0; ii<nn; ++ii) { 
  if ((flag==0)&&(bck[ii])) {
    kk1 = ii;
    kk2 = ii;
    flag= 1;
  } else {
  if ((flag)&&(bck[ii])) {
    kk2 = ii;
  } else {
  if ((flag)&&(bck[ii] == 0)) {
    abk1[nbk] = xx[kk1];
    abk2[nbk] = xx[kk2];
    if (ABS((abk1[nbk]-abk2[nbk])) > 0.3) { ++nbk; }
    if (nbk > MAXabk-3) { printf("***error: auto finder finds too many background regions.\n"); return(1); }
    flag = 0;
  }}}
}
if (flag) {
  kk2=nn-1;
  abk1[nbk] = xx[kk1];
  abk2[nbk] = xx[kk2];
  if (ABS((abk1[nbk]-abk2[nbk])) > 0.3) { ++nbk; }
  if (nbk > MAXabk-3) { printf("***error: auto finder finds too many background regions.\n"); return(1); }
}

printf("nbk=%d\n",nbk);
for (ii=0; ii<nbk; ++ii) {
  printf(" abk1=%8.4f  abk2=%8.4f \n",abk1[ii],abk2[ii]);
  
  fprintf(drawfu,"sls 3 ; sci 2\n");
  fprintf(drawfu,"%f %f\n",abk1[ii],0.);
  fprintf(drawfu,"%f %f\n",abk1[ii],999000.);
  fprintf(drawfu,"draw\n");
  fprintf(drawfu,"%f %f\n",abk2[ii],0.);
  fprintf(drawfu,"%f %f\n",abk2[ii],999000.);
  fprintf(drawfu,"draw\n");

}

/* Set background also, if requested. */
if (noback == 0) {
  SPX[0].nbk = nbk;
  for (ii=0; ii<nbk; ++ii) {
    SPX[0].abk1[ii] = abk1[ii];
    SPX[0].abk2[ii] = abk2[ii];
  }
}

fclose(drawfu);
return(0);
}



/* ----------------------------------------------------------------------
 Nightsky.  Centroid pixel position of known night sky lines.
*/
void nsx_nightsky( int nso, double nspbck_nob[9][3000], double xpos, double *cent, double *high, double *bck )
{
/**/
int kk,ii,ii1,ii2,nn,narr;
double wgt,sum,wsum,median,arr[40],xx8[40],yy8[40];
/* FILE *outfu; */
/**/
ii1 = cnint(xpos) - 9;
ii2 = cnint(xpos) + 9;
/* outfu = fopen_write("ns.dat"); */
/* fprintf(outfu,"| ii  | bck_nob |\n"); */
nn=0;
for (ii=ii1; ii<=ii2; ++ii) {
  xx8[nn]=(double)ii;
  yy8[nn]=nspbck_nob[nso][ii];
  ++nn;
/*  fprintf(outfu," %5d %9.3f \n",ii,nspbck_nob[nso][ii]); */
}
/* fclose(outfu); */
/* Centroid. */
narr=0; for (ii=0; ii<nn; ++ii) { arr[narr]=yy8[ii]; ++narr; }
median = cfind_median8(narr,arr);
kk = cnint( 0.1 * (double)narr );
*bck = arr[kk];
sum=0.; wsum=0.; *high=0.;
for (ii=6; ii<=12; ++ii) { 
  wgt = yy8[ii] - *bck;
  sum = sum + (xx8[ii] * wgt);
  wsum= wsum+ wgt;
  if (wgt > *high) *high=wgt;
}
if (wsum > 0.) { *cent = sum / wsum; } else { *cent=0.; }
return;
}


/* ----------------------------------------------------------------------
 Given x,y arrays and a new x point (xpt) return an interpolated value for
 the y axis.  x,y array must be monotonically increasing in x.  "0" is
 returned for xpt values outside of the range.
 Inputs:  n (number of points in x,y arrays).   x[],y[] (arrays).
          xpt (value to interpolate towards).
*/
/*@@*/
double nsx_yinterp(int n, double x[], double y[], double xpt)
{
/**/
  int ii;
  double yy;
/**/
  if (n   <   1   ) return(0.);
  if (xpt < x[0]  ) return(0.);
  if (xpt > x[n-1]) return(0.);
  ii=1;
  while ( xpt > x[ii] ) ++ii;
  yy = y[ii-1] + ( (y[ii]-y[ii-1]) * (xpt-x[ii-1]) / (x[ii]-x[ii-1]) );
  return(yy);
}


/* ----------------------------------------------------------------------
 Given B-V, return effective stellar temperature in degrees Kelvin
 (main sequence, solar metallicity).  Data from "Bowers and Deeming" text
 book and "Lejeune, Cuisinier, and Buser: A&ASupp 125, 229 (1997)."
*/
/*@@*/
double nsx_StarTempBmV(double userbv)
{
/**/
  double bv[27]=
{
-0.35 ,
-0.31 ,
-0.16 ,
-0.100,
-0.050,
+0.000,
+0.050,
+0.100,
+0.13 ,
+0.175,
+0.260,
+0.27 ,
+0.335,
+0.42 ,
+0.425,
+0.58 ,
+0.595,
+0.70 ,
+0.760,
+0.875,
+0.89 ,
+1.095,
+1.18 ,
+1.350,
+1.45 ,
+1.63 ,
+1.80
};
  double stp[27]=
{
 40000.,
 28000.,
 15500.,
 11750.,
 10200.,
  9500.,
  8870.,
  8600.,
  8500.,
  8000.,
  7500.,
  7400.,
  7000.,
  6580.,
  6500.,
  6030.,
  6000.,
  5520.,
  5500.,
  5000.,
  4900.,
  4500.,
  4130.,
  4000.,
  3480.,
  2800.,
  2400.
};
  double rr;
/**/
if (userbv <= bv[0] ) return(40000.);
if (userbv >= bv[26]) return(2400.);
rr = nsx_yinterp(27, bv, stp, userbv);
return(rr);
}





/* ----------------------------------------------------------------------
  Find Star in Tycho and find temperature. -tab 11aug2014 
*/
int nsx_StarTemp( TYCtype TYC[], int numTYC, double sra, double sdec, double *los, double *EffTemp )
{
/**/
int ok,ii,loii;
double losep,sep,diff,BmV;
/**/
losep=0.015; loii=-1; *EffTemp=-8.;
for (ii=0; ii<numTYC; ++ii) {
  diff = ABS((sdec - TYC[ii].mean_dec));
  if (diff < 0.015) {
    sep = cangsep(sra,sdec,TYC[ii].mean_ra,TYC[ii].mean_dec);
    if (sep < losep) {
      losep = sep; 
      loii  = ii;
/* Proper motion.
      pra = TYC[ii].mean_ra  + ( (TYC[ii].pm_ra  * 14.) / 3600000. );
      pdec= TYC[ii].mean_dec + ( (TYC[ii].pm_dec * 14.) / 3600000. );
      losep2 = cangsep(sra,sdec,pra,pdec);
*/
    }
  }
}
if (loii < 0) {
  printf("Could not find Tycho match.\n");
  ok=0;
} else {
  ok=1;
  if ((TYC[loii].bt_mag > -9.)&&(TYC[loii].vt_mag > -9.)) {
    BmV = TYC[loii].bt_mag - TYC[loii].vt_mag;
    *EffTemp = nsx_StarTempBmV( BmV );
  } else {
    BmV = -9.;
    *EffTemp = -9.;
  }
  printf("Found Tycho match, sep=%9.3f  tyra=%9.5f tydec=%9.5f EffTemp=%12.5f  BmV=%8.3f.\n",
          losep*3600.,TYC[loii].mean_ra,TYC[loii].mean_dec,*EffTemp,BmV);
}


*los = losep;
return(ok);
}



/* ----------------------------------------------------------------------
  Sky fit with rejections.  -tab 30may2014
*/
int nsx_skyfit( int nn8, double xx8[], double yy8[], double zz8[], int polyorder,
                double *bck_xoff, double bck_coef[], double siglim )
{
/**/
int totrej,nrej,iter,ii,ok;
double xv,ff,rms,num,dev;
/**/
ok = GJ_polyfit( nn8, xx8, yy8, zz8, polyorder, 0, bck_xoff, bck_coef );
if (siglim < 99.) {
  totrej=0; iter=0; nrej=99;
  while ((ok)&&(totrej < nn8/4)&&(iter < 4)&&(nrej > 0)) {
    rms=0.; num=0.;
    for (ii=0; ii<nn8; ++ii) { if (zz8[ii] > 0.) {
      xv = xx8[ii] - *bck_xoff;
      ff = cpolyval( polyorder+1, bck_coef, xv );
      rms = rms + ((yy8[ii] - ff) *(yy8[ii] - ff));
      num = num + 1.;
    }}
    rms = sqrt(( rms / num ));
    nrej=0;
    for (ii=0; ii<nn8; ++ii) { if (zz8[ii] > 0.) {
      xv = xx8[ii] - *bck_xoff;
      ff = cpolyval( polyorder+1, bck_coef, xv );
      dev = ABS((yy8[ii] - ff));
      if (dev > rms*siglim) { zz8[ii]=0.; ++nrej; ++totrej; }
    }}
    ok = GJ_polyfit( nn8, xx8, yy8, zz8, polyorder, 0, bck_xoff, bck_coef );
    ++iter;
  }
}

return(ok);
}


/* ----------------------------------------------------------------------
  Return 1 if NO absorption line.
*/
int nsx_NoAbsorption( double ww, int Van, double Va1[], double Va2[] )
{
/**/
int ii,NoA;
/**/
NoA = 1;
for (ii=0; ii<Van; ++ii) { if ((ww > Va1[ii])&&(ww < Va2[ii])) NoA=0; }
return(NoA);
}



/* ----------------------------------------------------------------------
  Vega matching..
*/
void nsx_vega( int nso, int Vn, double Vx[], double Vy[], int VCn, double VCx[], double VCy[],
               int Van, double Va1[], double Va2[], double VyD[] )
{
/**/
int ii,jj,order;
/**/
double rr,ww,mean,xmin,xmax,meanC,num;
double xv,xoff,coef[9],xx8[9000],yy8[9000],ww8[9000];
/**/
float xx4[9000],yy4[9000],ww4[9000],bb4[9000];
/**/

mean=0.; xmin=Vx[0]; xmax=Vx[0];
for (ii=0; ii<Vn; ++ii) { 
  mean = mean + Vy[ii];
  if (Vx[ii] < xmin) xmin=Vx[ii];
  if (Vx[ii] > xmax) xmax=Vx[ii];
}
mean = mean / (double)Vn;
xmin = xmin * 10000.;
xmax = xmax * 10000.;

meanC=0.; num=0.;
for (ii=0; ii<VCn; ++ii) { 
  if ((VCx[ii] > xmin)&&(VCx[ii] < xmax)) { meanC = meanC + VCy[ii];  num=num+1.; }
}
meanC = meanC / num;

printf("NOTE: mean=%12.5e  meanC=%12.5e  num=%f \n",mean,meanC,num);


/* Load values. */
for (ii=0; ii<Vn; ++ii) {
  ww = Vx[ii] * 10000.;
  jj = nsx_cneari( ww, VCn, VCx );
  rr = mean * VCy[jj] / meanC; 
  xx8[ii] = ww;
  yy8[ii] = rr;
  ww8[ii] = nsx_NoAbsorption( ww, Van, Va1, Va2 );
  xx4[ii] = (float)xx8[ii];
  yy4[ii] = (float)yy8[ii];
  ww4[ii] = (float)ww8[ii];
}


/* Fit Black Body. */
/*
nsx_FitBlackBody( Vn, xx4, yy4, ww4, &T, &a );
for (ii=0; ii<Vn; ++ii) { 
  bb4[ii] = nsx_BlackBodyAT( a, T, (double)xx4[ii] ); 
}
*/

/* Fit polynomial. */
order=4;
if (nso == 6) order=5;
if (GJ_polyfit(Vn,xx8,yy8,ww8,order,0,&xoff,coef) != 1) { printf("***error:tv:polyfit failed.\n"); exit(1); }
for (ii=0; ii<Vn; ++ii) { 
  xv = xx8[ii] - xoff;
  bb4[ii] = (float)cpolyval((order+1),coef,xv);
}

return;
}



/* ----------------------------------------------------------------------
 Load vega calibration spectrum.    -tab 13aug2014  -tab 06mar2018
*/
void nsx_load_vega( char nsxdir[], int *VCn, double VCx[], double VCy[], int VCmax )
{
/**/
char wrd[200];
char line[200];
/**/
double wave;
/**/
int nn;
/**/
FILE *infu;
/**/
nn=0;
sprintf(wrd,"%scal/VegaCal.tbl",nsxdir);
infu = fopen_read(wrd);
while (fgetline(line,infu)) { if (line[0] != '|') {
  wave = GLV(line,1);
  if ((wave > 8900.)&&(wave < 27000.)) {
    VCx[nn] = wave;
    VCy[nn] = GLV(line,2);
    ++nn;
    if (nn > VCmax-3) { printf("***error: too many vega points.\n"); exit(1); }
  }
}}
fclose(infu);
*VCn = nn;
return;
}




/* ----------------------------------------------------------------------
 Find Hot star for telluric correction. Read data which can then be applied
 to this spectrum.    -tab 13aug2014
 Returns '1' if found appropriate hot star for telluric correction.
 Vega is Teff=9600 (approx)..
*/
int nsx_find_hotstar( char nsxout[], char root[], NLStype NLS[], char HOTroot[] )
{
/**/
int kk,ii,vc,lc,pp,loii,lodif;
char wrd[200];
char wrd2[200];
char line[200];
double rr;
FILE *infu;
/**/
/* Find primary object. */
strcpy(HOTroot,"");
kk=-1;
for (ii=0; ii<NLS[0].num; ++ii) { if (strcmp(root,NLS[ii].root)==0) { kk=ii; } }
if (kk < 0) { printf("===warning: could not find this object in nsx.tbl file.\n"); return(0); }
/* Find appropriate hot star. */
lodif=9999999; loii=0;
for (ii=0; ii<NLS[0].num; ++ii) {  if (ii != kk) {
/* Right temperature? */
if ((NLS[ii].EffTemp > 9100.)&&(NLS[ii].EffTemp < 10100.)) {
/* Does spectrum exist? */
  sprintf(wrd,"%s%s-sp3.tbl",nsxout,NLS[ii].root);
  if (FileExist(wrd) == 1) {
/* Does it have VsumD values? */
    infu = fopen_read(wrd);
    vc=0; lc=0; pp=0;
    while ((fgetline(line,infu))&&(vc < 10)&&(lc < 100)) {
      if (line[0]=='|') { pp = cindex(line,"VsumD"); } else {
        if (pp > 0) {
          substrcpy_terminate(line,pp-1,clc(line),wrd2,0); rr=GLV(wrd2,1); if (rr > 0.) ++vc;
        }
      }
      ++lc;
    }
    fclose(infu);
    if (vc > 9) {
      if (ABS((kk-ii)) < lodif) { lodif= ABS((kk-ii)); loii = ii; }
    }
  }
}
}}
/* Echo. */
if (lodif > 999999) {
  printf("===warning: could not find appropriate hot star.\n");
} else {
  printf("Best match for atmospheric correction star is '%s' with Teff=%f ..\n",NLS[loii].root,NLS[loii].EffTemp);
  strcpy(HOTroot,NLS[loii].root);
}
return(1);
}



/* ----------------------------------------------------------------------
 Read in nsx.tbl if it exists.
 This file contains info on all NIRES image exposure files.
*/
void nsx_load_NLS( char nsxout[], NLStype NLS[] )
{
/**/
int kk;
char line[300];
char wrd[100];
FILE *infu;
/**/
NLS[0].num=0; 
sprintf(wrd,"%snsx.tbl",nsxout);
if (FileExist(wrd)) {
  infu = fopen_read(wrd);
  kk=0;
  while (fgetline(line,infu)) { 
    if (line[0] != '|') {
      substrcpy_terminate(line,108,111,NLS[kk].xtype,0);
      substrcpy_terminate(line,113,132,wrd,0); wrd[clc(wrd)+1]='\0'; strcpy(NLS[kk].root,wrd);
      NLS[kk].ra = cvalread0(line,36,44);
      NLS[kk].dec= cvalread0(line,46,54);
      NLS[kk].air= cvalread0(line,56,60);
      NLS[kk].exp= cvalread0(line,83,88);
      NLS[kk].EffTemp = cvalread0(line,140,148);
      substrcpy_terminate(line,1,19,wrd,0);
      NLS[kk].jd = misc_zulu_to_julian(wrd);
      ++kk;
      if (kk > MAXNLS-3) { printf("***error: too many entries in '%snsx.tbl' max is %d .\n",nsxout,MAXNLS); exit(1); }
    }
  }
  fclose(infu);
  NLS[0].num=kk;
}
return;
}



/* ----------------------------------------------------------------------
  Fractional pixel summation.  Sums up counts within boundaries.
  cb1,cb2 = column boundaries 1 and 2 (left and right).
  rb1,rb2 = row boundaries 1 and 2 (bottom and top).
  NOTE: You must check before calling that boundaries are within image[].
  NOTE: You must check before calling that boundaries are within image[].
  NOTE: You must check before calling that boundaries are within image[].
  NOTE: You must check before calling that boundaries are within image[].
*/
void nsx_fractional_pixel_2D( int nc, float image[], double cb1, double cb2, double rb1, double rb2, double *imgsum )
{
/**/
int ii,jj,pixno;
/**/
double colfrac,rowfrac,totfrac;
double pi1,pi2,pj1,pj2;
/**/
/* Look at all relevant pixels. */
*imgsum=0.;
for (ii=cnint(cb1); ii<=cnint(cb2); ++ii) {
for (jj=cnint(rb1); jj<=cnint(rb2); ++jj) {
/* Corners of pixel. */
  pi1=(double)ii - 0.5;  pi2=(double)ii + 0.5;
  pj1=(double)jj - 0.5;  pj2=(double)jj + 0.5;
/* Column enclosure. */
  if (pi1 > cb1) {
    if (pi2 < cb2) {
      colfrac = 1.0;           /* Fully enclosed. */
    } else {
      colfrac = (cb2 - pi1);   /* Right boundary in pixel. */
    }
  } else {
    if (pi2 < cb2) {
      colfrac = (pi2 - cb1);   /* Left boundary in pixel. */
    } else {
      colfrac = (cb2 - cb1);   /* Both boundaries within pixel. */
    }
  }
/* Row enclosure. */
  if (pj1 > rb1) {
    if (pj2 < rb2) {
      rowfrac = 1.0;           /* Fully enclosed. */
    } else {
      rowfrac = (rb2 - pj1);   /* Top boundary in pixel. */
    }
  } else {
    if (pj2 < rb2) {
      rowfrac = (pj2 - rb1);   /* Bottom boundary in pixel. */
    } else {
      rowfrac = (rb2 - rb1);   /* Both boundaries within pixel. */
    }
  }
  totfrac = colfrac * rowfrac;
  pixno = ii + (jj*nc);
  *imgsum = *imgsum + (totfrac * image[pixno]);
}}
return;
}



/* ----------------------------------------------------------------------
  Fractional pixel summation with real column boundaries (fixed row).  
  Sums up counts within boundaries.
  cb1,cb2 = column boundaries 1 and 2 (left and right).
  NOTE: You must check before calling that boundaries are within image[].
  NOTE: You must check before calling that boundaries are within image[].
  NOTE: You must check before calling that boundaries are within image[].
  NOTE: You must check before calling that boundaries are within image[].
*/
void nsx_fractional_pixel_cb( int nc, float image[], double cb1, double cb2, int jj, double *imgsum )
{
/**/
int ii,pixno;
/**/
double colfrac,pi1,pi2;
/**/
/* Look at all relevant pixels. */
*imgsum=0.;
for (ii=cnint(cb1); ii<=cnint(cb2); ++ii) {
/* Edges of pixel. */
  pi1=(double)ii - 0.5;  
  pi2=(double)ii + 0.5;
/* Column enclosure. */
  if (pi1 > cb1) {
    if (pi2 < cb2) {
      colfrac = 1.0;           /* Fully enclosed. */
    } else {
      colfrac = (cb2 - pi1);   /* Right boundary in pixel. */
    }
  } else {
    if (pi2 < cb2) {
      colfrac = (pi2 - cb1);   /* Left boundary in pixel. */
    } else {
      colfrac = (cb2 - cb1);   /* Both boundaries within pixel. */
    }
  }
  pixno = ii + (jj*nc);
  *imgsum = *imgsum + (colfrac * image[pixno]);
}
return;
}



/* ----------------------------------------------------------------------
  Fractional pixel summation with real row boundaries (fixed column).  
  Sums up counts within boundaries.
  rb1,rb2 = row boundaries 1 and 2 (lower and upper).
  NOTE: You must check before calling that boundaries are within image[].
  NOTE: You must check before calling that boundaries are within image[].
  NOTE: You must check before calling that boundaries are within image[].
  NOTE: You must check before calling that boundaries are within image[].
*/
double nsx_fractional_pixel_rb( int nc, float image[], double rb1, double rb2, int icol )
{
/**/
int jj,pixno;
/**/
double rowfrac,pi1,pi2,imgsum;
/**/
/* Look at all relevant pixels. */
imgsum=0.;
for (jj=cnint(rb1); jj<=cnint(rb2); ++jj) {
/* Edges of pixel. */
  pi1=(double)jj - 0.5;  
  pi2=(double)jj + 0.5;
/* Column enclosure. */
  if (pi1 > rb1) {
    if (pi2 < rb2) {
      rowfrac = 1.0;           /* Fully enclosed. */
    } else {
      rowfrac = (rb2 - pi1);   /* Upper boundary in pixel. */
    }
  } else {
    if (pi2 < rb2) {
      rowfrac = (pi2 - rb1);   /* Lower boundary in pixel. */
    } else {
      rowfrac = (rb2 - rb1);   /* Both boundaries within pixel. */
    }
  }
  pixno = icol + (jj*nc);
  imgsum = imgsum + (rowfrac * image[pixno]);
}
return(imgsum);
}



/* ----------------------------------------------------------------------
 Clear AVP Calibration Parameters.
*/
void nsx_clear_AVP( AVPtype AVP[] )
{
/**/
int ii,nso;
/**/
/* Each echelle order. */
for (nso=0; nso<9; ++nso) {
for (ii=0; ii<MAXSP; ++ii) {
  AVP[nso].xoff[ii]    = 0.;
  AVP[nso].coef[ii][0] = 0.;
  AVP[nso].coef[ii][1] = 0.;
  AVP[nso].coef[ii][2] = 0.;
}
}
return;
}


/* ----------------------------------------------------------------------
 Clear Wavelength Scale Calibration 
*/
void nsx_clear_WSC( )
{
/**/
int nso,ii;
/**/
/* Each echelle order. */
for (nso=0; nso<9; ++nso) {
  WSC[nso].order   = 0;
  WSC[nso].xoff    = 0.;
  WSC[nso].orderinv= 0;
  WSC[nso].xoffinv = 0.;
  for (ii=0; ii<9; ++ii) {
    WSC[nso].coef[ii]   = 0.;
    WSC[nso].coefinv[ii]= 0.;
  }
}
return;
}


/* ----------------------------------------------------------------------
 Clear NIRES Calibration Parameters.
*/
void nsx_clear_NCP( )
{
/**/
int nso,ii;
/**/
/* Each echelle order. */
for (nso=0; nso<9; ++nso) {
  NCP[nso].porda   = 0;
  NCP[nso].xoffa   = 0.;
  NCP[nso].pordb   = 0;
  NCP[nso].xoffb   = 0.;
  for (ii=0; ii<9; ++ii) {
    NCP[nso].coefa[ii]   = 0.;
    NCP[nso].coefb[ii]   = 0.;
  }
}
return;
}

/* ----------------------------------------------------------------------
 Clear image data.
*/
void nsx_clear_SPX( SPXtype SPX[] )
{
/**/
int nso,ii;
/**/
/* Each echelle order. */
for (nso=0; nso<9; ++nso) {
/* Windows. */
  SPX[nso].nsp = 0;
  SPX[nso].nbk = 0;
/* Profiles. */
  SPX[nso].numpro  = 0;
  SPX[nso].pro_apn = 0;
  SPX[nso].pro_cen = 0.;
  for (ii=0; ii<MAXPRO; ++ii) {
    SPX[nso].pro_apx[ii]   =0.;
    SPX[nso].pro_apymed[ii]=0.;
    SPX[nso].pro_apyave[ii]=0.;
  }
/* Spectra. */
  SPX[nso].numsp = 0;
  for (ii=0; ii<MAXSP; ++ii) {
    SPX[nso].spobj[ii]     =0.;
    SPX[nso].sperr[ii]     =0.;
    SPX[nso].spbck[ii]     =0.;
    SPX[nso].spsky[ii]     =0.;
    SPX[nso].spwav[ii]     =0.;
    SPX[nso].spdsp[ii]     =0.;
    SPX[nso].sprow[ii]     =0;
    SPX[nso].spatm[ii]     =0;
    SPX[nso].spoac[ii]     =0;
    SPX[nso].speac[ii]     =0;
  }
}
return;
}





/* ----------------------------------------------------------------------
  Listing.
*/ 
void nsx_listing( char listfile[], char imgfile[], char imgfile2[] )
{
/**/
FILE *infu;
FILE *infu2;
FILE *outfu;
/**/
char wrd[200];
char wrd1[80];
char wrd2[80];
char wrd3[80];
char wrd4[80];
char wrd5[80];
char wrd6[80];
char line[200];
char infile[200];
char ffile[200];
char datafile[100];
char object[100];
char obstype[100];
char targname[100];
char zulu[100];
/**/
double ra,dec,mjd,jd,hourang,airmass,exptime;
/**/
fitsfile *fptr;
/**/
int count,ii,hdutype,nc,nr;
int status = 0;
/**/

/* Set input file. */
if (cindex(imgfile,".ls") > 0) {
  strcpy(infile,imgfile);
} else {
  strcpy(infile,"nsx_temp.ls");
  outfu = fopen_write("nsx_temp.ls");
  if (cindex(imgfile,".fits") > 0) {
    if (FileExist(imgfile)) { fprintf(outfu,"%s\n",imgfile); }
  }
  if (cindex(imgfile2,".fits") > 0) {
    if (FileExist(imgfile2)) { fprintf(outfu,"%s\n",imgfile2); }
  }
  fclose(outfu);
}
printf("Reading '%s'.\n\n",infile);

/* Set output file unit. */
if (strcmp(listfile,"") == 0) {
  outfu = NULL;
} else {
if (strcmp(listfile,"stdout") == 0) {
  outfu = stdout;
} else {
  outfu = fopen_write(listfile);
  printf("Writing listing file '%s'.\n",listfile);
}}

/* Read input file(s). */
count=0;
infu = fopen_read(infile);
while (fgetline(ffile,infu) == 1) {
if (FileExist(ffile)&&(cindex(ffile,".fits") > 0)) {  
printf("ffile='%s'\n",ffile);
  fits_open_file( &fptr, ffile, READONLY, &status );
  if (status != 0) { printf("***error: problem reading '%s'.\n",ffile); exit(1); }
  fits_movabs_hdu( fptr, 1, &hdutype, &status );
  nc = cfua_inhead(fptr,"NAXIS1");
  nr = cfua_inhead(fptr,"NAXIS2");
  if ((nc != nc_Nominal)||(nr != nr_Nominal)) { printf("***error: unknown image size (%d x %d).\n",nc,nr); exit(1); }
  cfua_chead(fptr,"DATAFILE",datafile); misc_pad_blanks(datafile ,15,wrd1);
  cfua_chead(fptr,"OBJECT"  ,object  ); misc_pad_blanks(object   ,20,wrd2);
  cfua_chead(fptr,"TARGNAME",targname); misc_pad_blanks(targname ,16,wrd3);
  cfua_chead(fptr,"OBSTYPE" ,obstype ); misc_pad_blanks(obstype  ,12,wrd6);
  mjd     = cfua_fhead(fptr,"MJD-OBS");
  if (mjd > 0.) {
    jd = cfua_fhead(fptr,"MJD-OBS") + 2400000.5;
  } else { jd = 2086303.; }   /* 1000-01-01T12:00:00.000 */
  misc_julian_to_zulu(jd,zulu);

  cfua_chead(fptr,"RA" ,wrd); 
  ra = -99.;
  if ((clc(wrd) > 3)&&(cindex(wrd,":") > 0)) { ra =misc_hmss2deci(wrd,1); } else {
  if ( clc(wrd) > 2)                         { ra =GLV(wrd,1);            }      }

  cfua_chead(fptr,"DEC" ,wrd); 
  dec= -99.;
  if ((clc(wrd) > 3)&&(cindex(wrd,":") > 0)) { dec=misc_hmss2deci(wrd,0); } else {
  if ( clc(wrd) > 2)                         { dec=GLV(wrd,1);            }      }

  airmass = cfua_fhead(fptr,"AIRMASS");  if (airmass < -9.0) airmass=-9.;
  exptime = cfua_fhead(fptr,"ITIME");    if (exptime < -9.0) exptime=-9.;
  hourang = cfua_fhead(fptr,"HA");       if (hourang < -99.) hourang=-99.;
  fits_close_file( fptr, &status ); cfua_error(status);
  misc_pad_blanks(ffile,30,wrd4);

/* comments? */
  strcpy(wrd5,"...");
  if (FileExist("comments")) {
    infu2 = fopen_read("comments");
    while (fgetline(line,infu2)) {
      if (cindex(line,datafile) > -1) {
        ii = cindex(line,"#");
        substrcpy_terminate(line,ii+1,clc(line),wrd5,0);
      }
    }
    fclose(infu2);
  }

/* header? */
  if (count%30 == 0) {
    if (outfu != NULL) {
fprintf(outfu,"| UT                    | ra      | dec     |exptime |airmass| HA    | datafile      | object             | targname       | obstype    | fits_file                    |\n");
/*              2016-07-19T15:03:44.467 123456789 123456789 12345678 1234567 1234567 123456789012345 12345678901234567890 1234567890123456 123456789012 123456789012345678901234567890 */
    }
  }
  ++count;

  if (outfu != NULL) {
    fprintf(outfu," %s %9.5f %9.5f %8.2f %7.4f %7.3f %s %s %s %s %s %s \n",zulu,ra,dec,exptime,airmass,hourang,wrd1,wrd2,wrd3,wrd6,wrd4,wrd5);
  }

}}
fclose(infu);
if ((outfu != NULL)&&(outfu != stdout)) { fclose(outfu); }

return;
}


/* ----------------------------------------------------------------------
  Free the image arrays in IMG structure. 
*/
void nsx_free_image( IMGtype IMG[] )
{
free(IMG[0].image);
free(IMG[0].varimg);
free(IMG[0].clnimg);
free(IMG[0].corimg);
free(IMG[0].corimgNFD);
free(IMG[0].bckimg);
return;
}


/* ----------------------------------------------------------------------
  Clear IMG structure. 
*/
void nsx_clear_image( IMGtype IMG[] )
{
IMG[0].nc=0;
IMG[0].nr=0;
strcpy(IMG[0].file,"");
strcpy(IMG[0].root,"");
strcpy(IMG[0].utshut,"");
strcpy(IMG[0].object,"");
IMG[0].jd     = 0.;
IMG[0].ra     = 0.;
IMG[0].dec    = 0.;
IMG[0].airmass= 1.;
IMG[0].ha     = 0.;
IMG[0].az     = 0.;
IMG[0].el     = 0.;
IMG[0].parang = 0.;
IMG[0].rotposn= 0.;
IMG[0].exptime= 1.;
IMG[0].image  = NULL;
IMG[0].varimg = NULL;
IMG[0].clnimg = NULL;
IMG[0].corimg = NULL;
IMG[0].corimgNFD = NULL;
IMG[0].bckimg = NULL;
IMG[0].X      = 0;
return;
}


/* ----------------------------------------------------------------------
  Echo out the extraction windows.
*/
void nsx_echo_extraction_window( SPXtype SPX[], AVPtype AVP[], IMGtype IMG )
{
/**/
double rr1,rr2;
int ii;
/**/
printf(       "Number of object windows: %d \n",SPX[0].nsp);
fprintf(logfu,"Number of object windows: %d \n",SPX[0].nsp);
for (ii=0; ii<SPX[0].nsp; ++ii) {
  printf(       "Object window: %7.3f to %7.3f (arcsec) [flx=%6.1f  sigs=%6.2f  peak=%7.1f]\n",SPX[0].asp1[ii],SPX[0].asp2[ii],SPX[0].pflx[ii],SPX[0].sigs[ii],SPX[0].peak[ii]);
  fprintf(logfu,"Object window: %7.3f to %7.3f (arcsec) [flx=%6.1f  sigs=%6.2f  peak=%7.1f]\n",SPX[0].asp1[ii],SPX[0].asp2[ii],SPX[0].pflx[ii],SPX[0].sigs[ii],SPX[0].peak[ii]);
}

for (ii=0; ii<SPX[0].nsp; ++ii) {
  rr1 = nsx_AVPinv( AVP, 3, 1000, SPX[0].asp1[ii], IMG );
  rr2 = nsx_AVPinv( AVP, 3, 1000, SPX[0].asp2[ii], IMG );
  printf("Object window: %7.3f to %7.3f (pixels, echelle order 3, column 1000)\n",rr1,rr2);
}
printf(       "Number of background windows: %d \n",SPX[0].nbk);
fprintf(logfu,"Number of background windows: %d \n",SPX[0].nbk);
for (ii=0; ii<SPX[0].nbk; ++ii) {
  printf(       "Background window: %7.3f to %7.3f (arcsec)\n",SPX[0].abk1[ii],SPX[0].abk2[ii]);
  fprintf(logfu,"Background window: %7.3f to %7.3f (arcsec)\n",SPX[0].abk1[ii],SPX[0].abk2[ii]);
}
for (ii=0; ii<SPX[0].nbk; ++ii) {
  rr1 = nsx_AVPinv( AVP, 3, 1000, SPX[0].abk1[ii], IMG );
  rr2 = nsx_AVPinv( AVP, 3, 1000, SPX[0].abk2[ii], IMG );
  printf("Background window: %7.3f to %7.3f (pixels)\n",rr1,rr2);
}
return;
}



/* ----------------------------------------------------------------------
  Read NIRES object image file into structure.
*/
void nsx_read_image( IMGtype IMG[], int echo )
{
/**/
int jj,anynull,hdutype;
int status = 0;
/**/
char wrd[200];
/**/
double jd,mjd;
/**/
long nbuffer;
long firstpixel = 1;
/**/
float nullval = 0.;  /* don't check for null values in the image */
/**/
fitsfile *fptr;
/**/
/* Append .fits ? */
jj=cindex(IMG[0].file,".fits"); 
if (jj == -1) { strcat(IMG[0].file,".fits"); }
if (FileExist(IMG[0].file)) {
  IMG[0].X = 1;
  status=0;
  if (echo) {
    printf("Open and read FITS file '%s'.\n",IMG[0].file);
    fprintf(logfu,"Open and read FITS file '%s'.\n",IMG[0].file);
  }
  fits_open_file( &fptr, IMG[0].file, READONLY, &status );
  if (status != 0) { printf("***error: problem reading '%s'.\n",IMG[0].file); exit(1); }
  fits_movabs_hdu( fptr, 1, &hdutype, &status );
  IMG[0].nc = cfua_inhead(fptr,"NAXIS1");
  IMG[0].nr = cfua_inhead(fptr,"NAXIS2");
  if ((IMG[0].nc < 1800)||(IMG[0].nr < 900)) {
    printf("===NOTE: '%s' is not a valid spectral image.\n",IMG[0].file);
    exit(1);
  }

  jd = 2086303.;                  /* 1000-01-01T12:00:00.000 */
  mjd= cfua_fhead(fptr,"MJD-OBS");
  if (mjd > 0.) { jd = mjd + 2400000.5; }
  misc_julian_to_zulu(jd,IMG[0].utshut);
  IMG[0].jd = jd;

  cfua_chead(fptr,"OBJECT",IMG[0].object);  

  cfua_chead(fptr,"RA" ,wrd);
  IMG[0].ra = -99.;
  if ((clc(wrd) > 3)&&(cindex(wrd,":") > 0)) { IMG[0].ra =misc_hmss2deci(wrd,1); } else {
  if ( clc(wrd) > 2)                         { IMG[0].ra =GLV(wrd,1);            }      }

  cfua_chead(fptr,"DEC" ,wrd);
  IMG[0].dec= -99.;
  if ((clc(wrd) > 3)&&(cindex(wrd,":") > 0)) { IMG[0].dec=misc_hmss2deci(wrd,0); } else {
  if ( clc(wrd) > 2)                         { IMG[0].dec=GLV(wrd,1);            }      }

  IMG[0].airmass = cfua_fhead(fptr,"AIRMASS"); if (IMG[0].airmass <    0.) IMG[0].airmass=0.;
  IMG[0].ha      = cfua_fhead(fptr,"HA"     ); if (IMG[0].ha      < -400.) IMG[0].az     =0.;
  IMG[0].az      = cfua_fhead(fptr,"AZ"     ); if (IMG[0].az      < -400.) IMG[0].az     =0.;
  IMG[0].parang  = cfua_fhead(fptr,"PARANG" ); if (IMG[0].parang  < -400.) IMG[0].parang =0.;
  IMG[0].rotposn = cfua_fhead(fptr,"ROTPOSN"); if (IMG[0].rotposn < -400.) IMG[0].rotposn=0.;
  IMG[0].el      = cfua_fhead(fptr,"EL"     ); if ((IMG[0].el < 0.)||(IMG[0].el > 90.01)) IMG[0].el=0.;

  IMG[0].exptime = cfua_fhead(fptr,"EXPTIME"); 
  if (IMG[0].exptime < 0.) {
    IMG[0].exptime = cfua_fhead(fptr,"ITIME"); 
    if (IMG[0].exptime < 0.) IMG[0].exptime = 0.;
  }
  if (IMG[0].exptime < 1.) IMG[0].exptime = 1.;

  if (echo) {
    fprintf(logfu,"RA=%9.5f  DEC=%9.5f  Exp=%8.2f  Air=%6.3f  %s \n",
        IMG[0].ra,IMG[0].dec,IMG[0].exptime,IMG[0].airmass,IMG[0].utshut);
  }

  nbuffer = IMG[0].nc * IMG[0].nr;
  IMG[0].image = (float *)calloc((nbuffer+1000),sizeof(float));
  IMG[0].varimg= (float *)calloc((nbuffer+1000),sizeof(float));
  IMG[0].clnimg= (float *)calloc((nbuffer+1000),sizeof(float));
  IMG[0].corimg= (float *)calloc((nbuffer+1000),sizeof(float));
  IMG[0].corimgNFD= (float *)calloc((nbuffer+1000),sizeof(float));
  IMG[0].bckimg= (float *)calloc((nbuffer+1000),sizeof(float));
  fits_read_img( fptr, TFLOAT, firstpixel, nbuffer, &nullval, IMG[0].image, &anynull, &status ); cfua_error(status);
  fits_close_file( fptr, &status ); cfua_error(status);
}
return;
}


/* ----------------------------------------------------------------------
  Find average in image using inner core of image.
*/
double nsx_image_average( float image[], int nc, int nr )
{
/**/
int ii,jj,pixno,col1,col2,row1,row2;
double sum,num,average;
/**/
col1 = cnint( (float)nc * 0.25 ); col2 = cnint( (float)nc * 0.75 );
row1 = cnint( (float)nr * 0.25 ); row2 = cnint( (float)nr * 0.75 );
sum=0.; num=0.;
for (ii=col1; ii<=col2; ++ii) {
  for (jj=row1; jj<=row2; ++jj) {
    pixno = ii + (jj*nc);
    sum = sum + image[pixno];
    num = num + 1.;
  }
}
if (num < 1.) num=1.;
average = sum / num;
if (average > 999999.) average = 9999.;
if (average < -99999.) average = -999.;
return(average);
}


/* ----------------------------------------------------------------------
  Fit forward and reverse polynomials.
*/
void nsx_arcfit( int nntot, double pix[], double air[], int order, double *xoff, double coef[], double *xoffi, double coefi[] )
{
/**/
int ii,nn;
double xx[900],yy[900],ww[900];
double high,wrms,wppd;
/**/
/* Load. */
nn=0;
for (ii=0; ii<nntot; ++ii) {
  if (pix[ii] > 0.) { xx[nn] = pix[ii]; yy[nn] = air[ii]; ww[nn] = 1.0; ++nn; }
}
/* Forward [ wave = func(pix) ]. */
if (GJ_polyfit(nn,xx,yy,ww,order,0,xoff,coef) != 1) { printf("***error:: arcfit fit failed.\n"); exit(1); }
GJ_polyfit_residuals( nn, xx, yy, ww, order, *xoff, coef, &high, &wrms, &wppd );
printf("forward: high=%9.3f  wrms=%9.3f  wppd=%9.3f : order=%2d  nn=%4d  xoff=%12.5e  coef=%12.5e %12.5e %12.5e %12.5e \n",
      high,wrms,wppd,order,nn,*xoff,coef[0],coef[1],coef[2],coef[3]);
/* Inverse [pix = func(wave)]. */
if (GJ_polyfit(nn,yy,xx,ww,order,0,xoffi,coefi) != 1) { printf("***error:: inverse arcfit fit failed.\n"); exit(1); }
GJ_polyfit_residuals( nn, yy, xx, ww, order, *xoffi, coefi, &high, &wrms, &wppd );
printf("inverse: high=%9.3f  wrms=%9.3f  wppd=%9.3f : order=%2d  nn=%4d  xoff=%12.5e  coef=%12.5e %12.5e %12.5e %12.5e \n",
      high,wrms,wppd,order,nn,*xoffi,coefi[0],coefi[1],coefi[2],coefi[3]);
for (ii=order+1; ii<9; ++ii) { coef[ii]=0.; coefi[ii]=0.; }
return;
}


/* ----------------------------------------------------------------------
  Create and Write RMS spectra.
*/
void nsx_RMS_spectra( char root[], SPXtype SPX[], char nsxdir[], char nsxout[], char sfx[] )
{
/**/
int nso,ii,ii1,ii2,iii;
/**/
char wrd[100];
/**/
FILE *outfu;
/**/
double arr[3000],scr[3000],sprms[3000];
double rms,num;
/**/

/* Write spectrum file for each echelle order. */
for (nso=3; nso<=7; ++nso) {

  sprintf(wrd,"%s%s-rms%d%s.tbl",nsxout,root,nso,sfx);
  printf("Writing RMS spectrum table '%s'.\n",wrd);
  for (ii=0; ii<SPX[nso].numsp; ++ii) { arr[ii] = SPX[nso].spobj[ii];  sprms[ii]=0.; }
  nsx_SmoothArray8( 20, SPX[nso].numsp, arr, scr );

  for (ii=10; ii<SPX[nso].numsp-10; ++ii) {
    ii1=ii-10;
    ii2=ii+10;
    rms=0.; num=0.;
    for (iii=ii1; iii<=ii2; ++iii) {
      rms = rms + ( (SPX[nso].spobj[iii] - arr[iii]) * (SPX[nso].spobj[iii] - arr[iii]) );
      num = num + 1.;
    }
    if ((rms > 0.)&&(num > 0.)) { sprms[ii] = sqrt(( rms / num )); }
  }

  outfu = fopen_write(wrd);
  fprintf(outfu,"|  wave   | object     | error      | smooth     | rms        |\n");
/*                123456789 123456789012 123456789012 123456789012 123456789012  */
  for (ii=0; ii<SPX[nso].numsp; ++ii) {
    fprintf(outfu," %9.6f %12.5e %12.5e %12.5e %12.5e \n",
      SPX[nso].spwav[ii]/10000., SPX[nso].spobj[ii], SPX[nso].sperr[ii], arr[ii], sprms[ii] ); 
  }
  fclose(outfu);

}
return;
}



/* ----------------------------------------------------------------------
  Write spectra.
*/
void nsx_write_spectra( char root[], SPXtype SPX[], char nsxdir[], char nsxout[], char sfx[] )
{
/**/
int nso,ii;
/**/
char wrd[100];
/**/
FILE *outfu;
/**/
/* Write spectrum file for each echelle order. */
for (nso=3; nso<=7; ++nso) {
  sprintf(wrd,"%s%s-sp%d%s.tbl",nsxout,root,nso,sfx);
  printf("Writing spectrum table '%s'.\n",wrd);
  outfu = fopen_write(wrd);
  fprintf(outfu,"|  wave   | object     | error      | backgnd    | sky        | col | row   |angstrom | disp   | atmos  | obj_atmcor | err_atmcor |\n");
/*                123456789 123456789012 123456789012 123456789012 123456789012 12345 1234567 123456789 12345678 12345678 123456789012 123456789012 */
  for (ii=SPX[nso].numsp-1; ii>=0; --ii) {
    fprintf(outfu," %9.6f %12.5e %12.5e %12.5e %12.5e %5d %7.2f %9.3f %8.5f %8.6f %12.5e %12.5e \n",
      SPX[nso].spwav[ii]/10000., SPX[nso].spobj[ii], SPX[nso].sperr[ii], SPX[nso].spbck[ii], 
      SPX[nso].spsky[ii], ii, SPX[nso].sprow[ii], SPX[nso].spwav[ii], SPX[nso].spdsp[ii], 
      SPX[nso].spatm[ii], SPX[nso].spoac[ii], SPX[nso].speac[ii] );
  }
  fclose(outfu);

  /* KVGC-- Write as csv as well */
  sprintf(wrd,"%s%s-sp%d%s.csv",nsxout,root,nso,sfx);
  printf("KVGC : Writing spectrum table '%s'.\n",wrd);
  outfu = fopen_write(wrd);
  fprintf(outfu,"wave,object,error,backgnd,sky,col,row,angstrom,disp,atmos,obj_atmcor,err_atmcor\n");
/*                123456789 123456789012 123456789012 123456789012 123456789012 12345 1234567 123456789 12345678 12345678 123456789012 123456789012 */
  for (ii=SPX[nso].numsp-1; ii>=0; --ii) {
    fprintf(outfu,"%9.6f,%12.5e,%12.5e,%12.5e,%12.5e,%5d,%7.2f,%9.3f,%8.5f,%8.6f,%12.5e,%12.5e\n",
      SPX[nso].spwav[ii]/10000., SPX[nso].spobj[ii], SPX[nso].sperr[ii], SPX[nso].spbck[ii], 
      SPX[nso].spsky[ii], ii, SPX[nso].sprow[ii], SPX[nso].spwav[ii], SPX[nso].spdsp[ii], 
      SPX[nso].spatm[ii], SPX[nso].spoac[ii], SPX[nso].speac[ii] );
  }
  fclose(outfu);
  /* End write to csv file */


}
return;
}


/* ----------------------------------------------------------------------
  Compute fraction of a pixel is within given boundaries.
  pp1 is the lower real boundary position in pixel space. 
  pp2 is the upper real boundary position in pixel space. 
  rb1 is the real lower pixel edge (pixel-0.5).
  rb2 is the real upper pixel edge (pixel+0.5).
  (How much is rb1 to rb2 enclosed within pp1 to pp2 boundaries?)
*/
double nsx_boundary_fraction( double pp1, double pp2, double rb1, double rb2 )
{
/**/
double frac;
/**/
frac=0.;
if (rb1 > pp2) {                frac= 0.0;    /* fully above */
} else {
if (rb2 < pp1) {                frac= 0.0;    /* fully below */
} else {
if ((rb1 > pp1)&&(rb2 < pp2)) { frac= 1.0;    /* fully enclosed */
} else {
if ((rb1 < pp1)&&(pp2 > rb2)) { frac= (rb2 - pp1); /* pixel straddles lower boundary */
} else {
if ((rb2 > pp2)&&(pp1 < rb1)) { frac= (pp2 - rb1); /* pixel straddles upper boundary */
} else {
                                frac= (pp2 - pp1); /* boundaries must be within pixel */
}}}}}
return(frac);
}



/* ----------------------------------------------------------------------
  Write AmB profile (A - B)..   -tab 07feb2018
*/
void nsx_write_AmB_profile( char root[], AVPtype AVP[], SPXtype SPX[], SPXtype SPXB[], char nsxout[], char sfx[], IMGtype IMG )
{
/**/
int ii,kk;
/**/
char wrd[200];
/**/
double apymed,apyave,objfrac,bckfrac,rb1,rb2,rowoff;
/**/
FILE *outfu;
/**/

/* File name includes unique part of rootB.. */
sprintf(wrd,"%s%s-pro%s.tbl",nsxout,root,sfx); 

/* Combined over echelle orders profile in arcseconds. */
printf("Writing profile table '%s'.\n",wrd);
outfu = fopen_write(wrd);
fprintf(outfu,"| arcsec | median     | average    |obj |bck | row   |\n");
for (ii=0; ii<SPX[0].pro_apn; ++ii) {
  objfrac=0.;
  bckfrac=0.;
  if ((ii > 0)&&(ii < SPX[0].pro_apn-1)) {
    rb1 = (SPX[0].pro_apx[ii-1] + SPX[0].pro_apx[ii]) / 2.;
    rb2 = (SPX[0].pro_apx[ii] + SPX[0].pro_apx[ii+1]) / 2.;
    objfrac = nsx_boundary_fraction( SPX[0].asp1[0], SPX[0].asp2[0], rb1, rb2 );
    for (kk=0; kk<SPX[0].nbk; ++kk) {
      bckfrac = bckfrac + nsx_boundary_fraction( SPX[0].abk1[kk], SPX[0].abk2[kk], rb1, rb2 );
    }
  }
  rowoff = nsx_AVPinv( AVP, 3, 1000, SPX[0].pro_apx[ii], IMG );  
  apymed = SPX[0].pro_apymed[ii] - SPXB[0].pro_apymed[ii];
  apyave = SPX[0].pro_apyave[ii] - SPXB[0].pro_apyave[ii];
  fprintf(outfu," %8.3f %12.3f %12.3f %4.2f %4.2f %7.3f \n", SPX[0].pro_apx[ii], 
            apymed, apyave, objfrac, bckfrac, rowoff );
}
fclose(outfu);

return;
}



/* ----------------------------------------------------------------------
  Write profiles.
*/
void nsx_write_profiles( char root[], AVPtype AVP[], SPXtype SPX[], char nsxout[], char sfx[], IMGtype IMG )
{
/**/
int ii,nso,kk;
/**/
char wrd[100];
/**/
double objfrac,bckfrac,rb1,rb2,rowoff;
/**/
FILE *outfu;
/**/

/* Combined over echelle orders profile in arcseconds. */
sprintf(wrd,"%s%s-pro%s.tbl",nsxout,root,sfx); 
printf("Writing profile table '%s'.\n",wrd);
outfu = fopen_write(wrd);
fprintf(outfu,"| arcsec | median     | average    |obj |bck | row   |\n");
for (ii=0; ii<SPX[0].pro_apn; ++ii) {
  objfrac=0.;
  bckfrac=0.;
  if ((ii > 0)&&(ii < SPX[0].pro_apn-1)) {
    rb1 = (SPX[0].pro_apx[ii-1] + SPX[0].pro_apx[ii]) / 2.;
    rb2 = (SPX[0].pro_apx[ii] + SPX[0].pro_apx[ii+1]) / 2.;
    objfrac = nsx_boundary_fraction( SPX[0].asp1[0], SPX[0].asp2[0], rb1, rb2 );
    for (kk=0; kk<SPX[0].nbk; ++kk) {
      bckfrac = bckfrac + nsx_boundary_fraction( SPX[0].abk1[kk], SPX[0].abk2[kk], rb1, rb2 );
    }
  }
  rowoff = nsx_AVPinv( AVP, 3, 1000, SPX[0].pro_apx[ii], IMG );  
  fprintf(outfu," %8.3f %12.3f %12.3f %4.2f %4.2f %7.3f \n", SPX[0].pro_apx[ii], 
            SPX[0].pro_apymed[ii], SPX[0].pro_apyave[ii], objfrac, bckfrac, rowoff );
}
fclose(outfu);

/* Each echelle order profiles in arcseconds. */
for (nso=3; nso<=7; ++nso) {
  sprintf(wrd,"%s%s-pro%d%s.tbl",nsxout,root,nso,sfx);
  printf("Writing profile table '%s'.\n",wrd);
  outfu = fopen_write(wrd);
  fprintf(outfu,"| arcsec | median     | average    |obj |bck | row   |\n");
  for (ii=0; ii<SPX[nso].pro_apn; ++ii) {
    objfrac=0.;
    bckfrac=0.;
    if ((ii > 0)&&(ii < SPX[nso].pro_apn-1)) {
      rb1 = (SPX[nso].pro_apx[ii-1] + SPX[nso].pro_apx[ii]) / 2.;
      rb2 = (SPX[nso].pro_apx[ii] + SPX[nso].pro_apx[ii+1]) / 2.;
      objfrac = nsx_boundary_fraction( SPX[0].asp1[0], SPX[0].asp2[0], rb1, rb2 );
      for (kk=0; kk<SPX[0].nbk; ++kk) {
        bckfrac = bckfrac + nsx_boundary_fraction( SPX[0].abk1[kk], SPX[0].abk2[kk], rb1, rb2 );
      }
    }
    rowoff = nsx_AVPinv( AVP, nso, 1000, SPX[0].pro_apx[ii], IMG );  
    fprintf(outfu," %8.3f %12.3f %12.3f %4.2f %4.2f %7.3f \n", SPX[nso].pro_apx[ii],
              SPX[nso].pro_apymed[ii], SPX[nso].pro_apyave[ii], objfrac, bckfrac, rowoff );
  }
  fclose(outfu);
}

return;
}




/* ----------------------------------------------------------------------
  Find offset value (in pixels) for a given rowoff (and nso and column).
*/
double nsx_SOP1( SOPtype SOP1[], int nso, int icol, double rowoff )
{
/**/
double xv,offset;
int order = 2;
/**/
xv = rowoff - SOP1[nso].xoff[icol];
offset = cpolyval( order+1, SOP1[nso].coef[icol], xv );
return(offset);
}

/* ----------------------------------------------------------------------
  Find offset value (in pixels) for a given rowoff (and nso and column).
  Use both SOP1[] and SOP2[] ..
*/
double nsx_SOP2( SOPtype SOP1[], SOPtype SOP2[], int nso, int icol, double rowoff )
{
/**/
double xv,offset,offset1,offset2;
int order = 2;
/**/
xv = rowoff - SOP1[nso].xoff[icol];
offset1 = cpolyval( order+1, SOP1[nso].coef[icol], xv );
xv = rowoff - SOP2[nso].xoff[icol];
offset2 = cpolyval( order+1, SOP2[nso].coef[icol], xv );
offset = offset1 + offset2;
return(offset);
}


/* ----------------------------------------------------------------------
  Compute slant column boundaries for extraction using SOP1[] --tab 05jan2018. 
*/
void nsx_slant_boundaries_SOP1( int nso, int icol, double rowoff, SOPtype SOP1[], int ecol, double *cb1, double *cb2 )
{
/**/
double col1,col2,col0,offset,offset0;
/**/
offset0 = nsx_SOP1( SOP1, nso, icol, rowoff );
col0 = (double)icol + offset0;
if ((icol-1) >=  0 ) { offset = nsx_SOP1( SOP1, nso, icol-1, rowoff ); } else { offset = offset0; }
col1 = (double)(icol-1) + offset;
if ((icol+1) < ecol) { offset = nsx_SOP1( SOP1, nso, icol+1, rowoff ); } else { offset = offset0; }
col2 = (double)(icol+1) + offset;
*cb1 = (col0 + col1) / 2.;
*cb2 = (col0 + col2) / 2.;
if (*cb1 < 0.) *cb1 = 0.;
if (*cb2 > (double)(ecol-1)) *cb2 = (double)(ecol-1);
return;
}


/* ----------------------------------------------------------------------
  Compute slant column boundaries for extraction using SOP1[] and SOP2[] .. --tab 16jan2018. 
*/
void nsx_slant_boundaries_SOP2( int nso, int icol, double rowoff, SOPtype SOP1[], SOPtype SOP2[], int ecol, double *cb1, double *cb2 )
{
/**/
double col1,col2,col0,offset,offset0;
/**/
if ((icol  ) >=  0 ) { offset = nsx_SOP2( SOP1, SOP2, nso, icol  , rowoff ); } else { offset = 0.; }
col0 = (double)(icol  ) + offset;
offset0 = offset;
if ((icol-1) >=  0 ) { offset = nsx_SOP2( SOP1, SOP2, nso, icol-1, rowoff ); } else { offset = offset0; }
col1 = (double)(icol-1) + offset;
if ((icol+1) < ecol) { offset = nsx_SOP2( SOP1, SOP2, nso, icol+1, rowoff ); } else { offset = offset0; }
col2 = (double)(icol+1) + offset;
*cb1 = (col0 + col1) / 2.;
*cb2 = (col0 + col2) / 2.;
if (*cb1 < 0.) *cb1 = 0.;
if (*cb2 > (double)(ecol-1)) *cb2 = (double)(ecol-1);
return;
}



/* ----------------------------------------------------------------------
 SKYLINE adjustment -tab 11jan2018 ..
   Compute slant offset correction polynomials from offimg2[] image.
   Write out SOP2.dat ..  Load offimg3[] ..
*/
void nsx_SOP2CALIB( float offimg2[], float offimg3[], int nc, int nr, SPXtype SPX[] )
{
/**/
int jjoff,irow,ecol,order,pixno,ii,nn,nso,icol;
/**/
double xx[300],yy[300],ww[300];
double xv,ff,coef[9],xoff,high,wrms,wppd;
/**/
FILE *outfu;
/**/
SOPtype SOP2[9];
/**/
for (nso=3; nso<=7; ++nso) { if (nso > 0) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  jjoff = ((7-nso) * 200);
  for (icol=0; icol<ecol; ++icol) {
    nn=0;
    for (irow=jjoff; irow<(jjoff+SPX[nso].numpro); irow=irow+1) {
      xx[nn]= (double)(irow - jjoff);
      pixno = icol + (irow * nc);
      yy[nn]= offimg2[pixno];
      ww[nn]= 1.0;
      ++nn;
    }
    order = 2;
    if (GJ_polyfit(nn,xx,yy,ww,order,0,&xoff,coef) != 1) { printf("***error:sop2 fit failed.\n"); exit(1); }
    GJ_polyfit_residuals( nn, xx, yy, ww, order, xoff, coef, &high, &wrms, &wppd );
    if ((ABS((high)) > 0.25)||(icol > 9999)) {
      printf("for:nso=%d  icol=%5d high=%8.4f wrms=%8.4f wppd=%8.4f\n",nso,icol,high,wrms,wppd);
      outfu = fopen_write("tst.dat");
      for (ii=0; ii<nn; ++ii) {
        ff = cpolyval((order+1),coef,(xx[ii]-xoff));
        fprintf(outfu," %9.4f %9.4f %9.4f %9.4f \n",xx[ii],yy[ii],ff,yy[ii]-ff);
      }
      fclose(outfu);
      printf("look at tst.dat..\n");
      cpauseit();
    }
    SOP2[nso].xoff[icol]    = xoff;
    SOP2[nso].coef[icol][0] = coef[0];
    SOP2[nso].coef[icol][1] = coef[1];
    SOP2[nso].coef[icol][2] = coef[2];

/* Load offimg3[] .. */
    for (irow=jjoff; irow<(jjoff+SPX[nso].numpro); irow=irow+1) {
      pixno = icol + (irow * nc);
      xv    = (double)(irow - jjoff);
      ff = cpolyval((order+1),coef,(xv-xoff));
      offimg3[pixno] = ff;
    }
  }
}}

/* Write out SOP2.dat .. */
outfu = fopen_write("SOP2.dat");
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (icol=0; icol<ecol; ++icol) {
    fprintf(outfu," %3d %4d %20.12e %20.12e %20.12e %20.12e \n",nso,icol,
            SOP2[nso].xoff[icol], SOP2[nso].coef[icol][0], SOP2[nso].coef[icol][1], SOP2[nso].coef[icol][2]);
  }
}
fclose(outfu);

return;
}




/* ----------------------------------------------------------------------
 Compute slant column boundaries for extraction.
*/
void nsx_slant_boundaries( int ii, int pp, SLTtype SLT, int ecol, double *cb1, double *cb2 )
{
/**/
int px;
double col0,col1,col2,offset,offset0;
/**/
/* Center. */
px = ii + (pp*SLT.nc);
offset0 = SLT.image[px];
if (offset0 < -9.) { printf("***error:1: bad slant offset.. should not happen [%f].\n",offset0); exit(1); }
col0 = (double)ii - offset0;
/* Left. */
offset = offset0;
if ((ii-1) >= 0) {
  px = (ii-1) + (pp*SLT.nc);
  offset = SLT.image[px];
  if (offset < -9.) { printf("***error:2: bad slant offset.. should not happen [%f].\n",offset); exit(1); }
}
col1 = (double)(ii-1) - offset;
/* Right. */
offset = offset0;
if ((ii+1) < ecol) {
  px = (ii+1) + (pp*SLT.nc);
  offset = SLT.image[px];
  if (offset < -9.) { printf("***error:3: bad slant offset.. should not happen [%f] (%d %d).\n",offset,ii,pp); exit(1); }
}
col2 = (double)(ii+1) - offset;
*cb1 = (col0 + col1) / 2.;
*cb2 = (col0 + col2) / 2.;
if (*cb1 < 0.) *cb1 = 0.;
if (*cb2 > (double)(ecol-1)) *cb2 = (double)(ecol-1);
return;
}

/* ----------------------------------------------------------------------
 Compute slant column boundaries for extraction.
*/
void nsx_slant_boundaries_real( int ii, double urpp, SLTtype SLT, int numpro, int ecol, double *cb1, double *cb2 )
{
/**/
int ipp,ipp1,ipp2;
double rpp,pp,fp1,fp2,cb1_1,cb1_2,cb2_1,cb2_2;
/**/

/* Check. */
rpp = urpp;
if (rpp < 0.) rpp=0.;
if (rpp > (double)(numpro-1)) rpp = (double)(numpro-1);

/* Set. */
ipp= cnint(rpp);
pp = (double)ipp;
ipp1=ipp; fp1=1.0; ipp2=ipp; fp2=0.0;

/* Pixel weights. */
if (rpp >= pp) {
  ipp1= ipp;
  fp1 = 1.0 - (rpp - pp);
  if (fp1 < 0.0) fp1=0.0;
  ipp2= ipp+1;
  fp2 = 1.0 - fp1;
}
if (rpp < pp) {
  ipp2= ipp;
  fp2 = 1.0 - (pp - rpp);
  if (fp2 < 0.0) fp2=0.0;
  ipp1= ipp-1;
  fp1 = 1.0 - fp2;
}

/* Compute weighted boundaries. */
nsx_slant_boundaries( ii, ipp1, SLT, ecol, &cb1_1, &cb2_1 );
nsx_slant_boundaries( ii, ipp2, SLT, ecol, &cb1_2, &cb2_2 );
*cb1 = (fp1 * cb1_1) + (fp2 * cb1_2);
*cb2 = (fp1 * cb2_1) + (fp2 * cb2_2);
return;
}



/* ----------------------------------------------------------------------
   Compute slant polynomials from old style SLT[] values.
   Must be done once in order to compute SOP1 (slant offset polynomials).
   Write out SOP1.dat and SOPinv.dat ..  -tab 05jan2018
*/
void nsx_SOP1CALIB( SLTtype SLT[], SPXtype SPX[], int nc )
{
/**/
int ecol,order,ii,nn,nso,icol;
/**/
double edge,cba,cb1,cb2,offset,rowoffmax;
double xx[300],yy[300],ww[300];
double rpp,ff,coef[9],xoff,high,wrms,wppd;
/**/
FILE *outfu;
FILE *outfu2;
/**/
SOPtype SOP1[9];
SOPItype SOPI[9];
/**/

outfu = fopen_write("sop1_test.dat");
for (nso=3; nso<=7; ++nso) { if (nso > 0) {
  rowoffmax = (double)SPX[nso].numpro - 0.1;
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (icol=10; icol<ecol-2; ++icol) {
    edge = nsx_find_real_image_row( 1, icol, nso );
    nn=0;
    for (rpp=0.; rpp<rowoffmax; rpp=rpp+1.) {
      nsx_slant_boundaries_real( icol, rpp, SLT[nso], SPX[nso].numpro, ecol, &cb1, &cb2 );
      cba = (cb1 + cb2) / 2.;
      offset = cba - (double)icol;
      fprintf(outfu,"%5d %8.2f %9.4f\n",icol,rpp,offset);
      xx[nn] = rpp;
      yy[nn] = offset;
      ww[nn] = 1.0;
      ++nn;
    }

    order = 2;
    if (GJ_polyfit(nn,xx,yy,ww,order,0,&xoff,coef) != 1) { printf("***error:sop1 fit failed.\n"); exit(1); }
    GJ_polyfit_residuals( nn, xx, yy, ww, order, xoff, coef, &high, &wrms, &wppd );
    if ((high > 0.0001)||(icol > 2099)) {
      printf("for:nso=%d  icol=%5d high=%20.12e wrms=%20.12e wppd=%20.12e\n",nso,icol,high,wrms,wppd);
      outfu2 = fopen_write("tst.dat");
      for (ii=0; ii<nn; ++ii) {
        ff = cpolyval((order+1),coef,(xx[ii]-xoff));
        fprintf(outfu2," %9.4f %9.4f %9.4f %9.4f \n",xx[ii],yy[ii],ff,yy[ii]-ff);
      }
      fclose(outfu2);
      printf("look at tst.dat..\n");
      cpauseit();
    }
    SOP1[nso].xoff[icol]    = xoff;
    SOP1[nso].coef[icol][0] = coef[0];
    SOP1[nso].coef[icol][1] = coef[1];
    SOP1[nso].coef[icol][2] = coef[2];

    order = 3;
    if (GJ_polyfit(nn,yy,xx,ww,order,0,&xoff,coef) != 1) { printf("***error:inv sopi fit failed.\n"); exit(1); }
    GJ_polyfit_residuals( nn, yy, xx, ww, order, xoff, coef, &high, &wrms, &wppd );
    if (high > 0.168) {
      printf("inv:nso=%d  icol=%5d high=%20.12e wrms=%20.12e wppd=%20.12e\n",nso,icol,high,wrms,wppd);
      outfu2 = fopen_write("tst.dat");
      for (ii=0; ii<nn; ++ii) {
        ff = cpolyval((order+1),coef,(yy[ii]-xoff));
        fprintf(outfu2," %9.4f %9.4f %9.4f %9.4f \n",yy[ii],xx[ii],ff,xx[ii]-ff);
      }
      fclose(outfu2);
      printf("look at tst.dat..\n");
      cpauseit();
    }
    SOPI[nso].xoffinv[icol]    = xoff;
    SOPI[nso].coefinv[icol][0] = coef[0];
    SOPI[nso].coefinv[icol][1] = coef[1];
    SOPI[nso].coefinv[icol][2] = coef[2];
    SOPI[nso].coefinv[icol][3] = coef[3];

  }
}}
fclose(outfu);

/* Copy data for first 10 columns. */
for (nso=3; nso<=7; ++nso) {
  ii = 10;
  for (icol=0; icol<10; ++icol) {
    SOP1[nso].xoff[icol]      = SOP1[nso].xoff[ii];
    SOP1[nso].coef[icol][0]   = SOP1[nso].coef[ii][0];
    SOP1[nso].coef[icol][1]   = SOP1[nso].coef[ii][1];
    SOP1[nso].coef[icol][2]   = SOP1[nso].coef[ii][2];

    SOPI[nso].xoffinv[icol]   = SOPI[nso].xoffinv[ii];
    SOPI[nso].coefinv[icol][0]= SOPI[nso].coefinv[ii][0];
    SOPI[nso].coefinv[icol][1]= SOPI[nso].coefinv[ii][1];
    SOPI[nso].coefinv[icol][2]= SOPI[nso].coefinv[ii][2];
    SOPI[nso].coefinv[icol][3]= SOPI[nso].coefinv[ii][3];
  }
}

/* Copy data for last 2 columns. */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  ii = ecol-3;
  for (icol=ecol-2; icol<ecol; ++icol) {
    SOP1[nso].xoff[icol]      = SOP1[nso].xoff[ii];
    SOP1[nso].coef[icol][0]   = SOP1[nso].coef[ii][0];
    SOP1[nso].coef[icol][1]   = SOP1[nso].coef[ii][1];
    SOP1[nso].coef[icol][2]   = SOP1[nso].coef[ii][2];

    SOPI[nso].xoffinv[icol]   = SOPI[nso].xoffinv[ii];
    SOPI[nso].coefinv[icol][0]= SOPI[nso].coefinv[ii][0];
    SOPI[nso].coefinv[icol][1]= SOPI[nso].coefinv[ii][1];
    SOPI[nso].coefinv[icol][2]= SOPI[nso].coefinv[ii][2];
    SOPI[nso].coefinv[icol][3]= SOPI[nso].coefinv[ii][3];
  }
}

/* Write out SOP1.dat and SOPinv.dat .. */
outfu = fopen_write("SOP1.dat");
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (icol=0; icol<ecol; ++icol) {
    fprintf(outfu," %3d %4d %20.12e %20.12e %20.12e %20.12e \n",nso,icol,
            SOP1[nso].xoff[icol], SOP1[nso].coef[icol][0], SOP1[nso].coef[icol][1], SOP1[nso].coef[icol][2]);
  }
}
fclose(outfu);
outfu = fopen_write("SOPinv.dat");
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (icol=0; icol<ecol; ++icol) {
    fprintf(outfu," %3d %4d %20.12e %20.12e %20.12e %20.12e %20.12e \n",nso,icol,
            SOPI[nso].xoffinv[icol], SOPI[nso].coefinv[icol][0], SOPI[nso].coefinv[icol][1], 
            SOPI[nso].coefinv[icol][2], SOPI[nso].coefinv[icol][3] );
  }
}
fclose(outfu);

return;
}




/* ----------------------------------------------------------------------
  Create a sigma image for a subsection of an image.
  Set siglim high (99999.) for first pass, then low (5.) for second pass.
*/
void nsx_sigma_subsection( int nc, int nr, float image[], float sigimg[], float valimg[],
                           int c1, int c2, int r1, int r2, int iipix, int jjpix, float siglim )
{
/**/
int kk,ii,jj,pixno,narr;
/**/
int mp = (1+c2-c1) * (1+r2-r1);
float arr[mp];
float median,rms;
/**/

/* Median and RMS from median. */
narr=0;
for (ii=c1; ii<=c2; ++ii) {
for (jj=r1; jj<=r2; ++jj) {
if ((ii != iipix)&&(jj != jjpix)) {
  pixno = ii + (jj * nc);
  if (sigimg[pixno] < siglim) {
    arr[narr] = image[pixno];
    ++narr;
  }
}}}
median = cfind_median(narr,arr);
rms=0.;
for (kk=0; kk<narr; ++kk) { rms = rms + ((arr[kk] - median) * (arr[kk] - median)); }
rms = sqrt(( rms / (double)narr ));

/* Load sigmas image. */
pixno = iipix + (jjpix * nc);
valimg[pixno] = image[pixno] - median;
sigimg[pixno] = ABS((valimg[pixno])) / rms;

return;
}



/* ----------------------------------------------------------------------
  Return median near a pixel.  Avoid flgimg[] pixels >= 0.001 .
  ii,jj = pixel location
  colrad, rowrad = column and row radius (integers)
  arr[] = scratch array big enough for all pixels in median box (up to maxarr).
*/
float nsx_median_box( float image[], float flgimg[], int nc, int nr, int ii, int jj,
                      int colrad, int rowrad, float arr[], int maxarr ) 
{
/**/
int c1,c2,r1,r2,iii,jjj,ppp,narr;
float median;
/**/
c1 = ii-colrad;   if (c1 < 0   ) c1=0;
c2 = ii+colrad;   if (c2 > nc-1) c2=nc-1;
r1 = jj-rowrad;   if (r1 < 0   ) r1=0;
r2 = jj+rowrad;   if (r2 > nr-1) r2=nr-1;
narr=0;
for (iii=c1; iii<=c2; ++iii) {
for (jjj=r1; jjj<=r2; ++jjj) {
  ppp = iii + (jjj * nc);
  if (flgimg[ppp] < 0.001) { 
    arr[narr] = image[ppp]; 
    ++narr; 
    if (narr > maxarr-3) { printf("***error: median box too large.\n"); exit(1); }
  }
}}
median = cfind_median(narr,arr);
return(median);
}


/* ----------------------------------------------------------------------
  Cleaning image (on uncorrected image).
*/
void nsx_clean_image( IMGtype IMG[], SPXtype SPX[], int NoClean, int NoHotClean, char nsxdir[] )
{
/**/
int nbad,ecol,pixno,ii,jj,c1,c2,r1,r2;
int ff,bb,nr,nc,nso,col1,col2,row1,row2,ppp,iii,jjj;
/**/
float edge,maxsig,adjsig;
/**/
int flgcount=0;
/**/
float *flgimg;
float *valimg;
float *sigimg;
float siglim;
/**/
char root[100];
char wrd[200];
char line[200];
char badfile[2][200];
/**/
const int maxbad = 6000;
int bii[maxbad],bjj[maxbad],biirad[maxbad],bjjrad[maxbad];
/**/
const int maxarr = 1000;
float arr[maxarr];
/**/
FILE *infu;
/**/

/* Set */
strcpy(root,IMG[0].root); nc=IMG[0].nc; nr=IMG[0].nr;

/* Check image size. */
if ((nc != nc_Nominal)||(nr != nr_Nominal)) {
  printf("***error: wrong image dimensions (%d %d).\n",nc,nr);
  exit(1);
}

/* Copy. */
for (ii=0; ii<(nc*nr); ++ii) { IMG[0].clnimg[ii] = IMG[0].image[ii]; }

/* Allocate. */
flgimg = (float *)calloc((nc*nr),sizeof(float));

/* Clean bad pixels. */
if (NoHotClean == 0) {

  nbad=0;
  sprintf(badfile[0],"%s/cal/HotPix.fits",nsxdir);
  sprintf(badfile[1],"%s/cal/LowPix.fits",nsxdir);
  for (ff=0; ff<2; ++ff) { 
    nsx_read_general_image( badfile[ff], flgimg, nc, nr );
    for (ii=0; ii<nc; ++ii) {
    for (jj=0; jj<nr; ++jj) {
      pixno = ii + (jj * nc);
      if (flgimg[pixno] > 0.1) {
        bii[nbad] = ii;
        bjj[nbad] = jj;
        biirad[nbad] = 1;
        bjjrad[nbad] = 4;
        ++nbad;
        if (nbad > maxbad-3) { printf("***error: too many bad pixels from calibration.\n"); exit(1); }
      }
    }}
  }
  sprintf(wrd,"%s/cal/BadPix.dat",nsxdir);
  infu = fopen_read(wrd);
  while (fgetline(line,infu)) {
    bii[nbad] = GLV(line,1);
    bjj[nbad] = GLV(line,2);
    biirad[nbad] = GLV(line,3);
    bjjrad[nbad] = GLV(line,4);
    ++nbad;
    if (nbad > maxbad-3) { printf("***error: too many bad pixels from calibration.\n"); exit(1); }
  }
  fclose(infu);
  printf("Read %d bad pixels from calibration.\n",nbad);

/* Bad image flags. */
  for (ii=0; ii<(nc*nr); ++ii) { flgimg[ii]=0.; }
  for (bb=0; bb<nbad; ++bb) {
    pixno = bii[bb] + (nc * bjj[bb]);
    flgimg[pixno] = 1.;
  }

/* Replace known bad pixels with local median. */
  for (bb=0; bb<nbad; ++bb) {
    pixno = bii[bb] + (nc * bjj[bb]);
    IMG[0].clnimg[pixno] = nsx_median_box(IMG[0].image,flgimg,nc,nr,bii[bb],bjj[bb],biirad[bb],bjjrad[bb],arr,maxarr);
  }

/* #@# */
  nsx_write_general_image( "flgimg0.fits", flgimg, nc, nr );
/* #@# */

}

/* No more if 'NoClean' set. */
if (NoClean) { free(flgimg); return; }

/* Alloc */
valimg = (float *)calloc((nc*nr),sizeof(float));
sigimg = (float *)calloc((nc*nr),sizeof(float));

/* Clear. */
for (ii=0; ii<(nr*nc); ++ii) { flgimg[ii]=0.; valimg[ii]=0.; sigimg[ii]=0.; }

/* Sigma image for all nso within slit (using hot pixel corrected image).. */
/* First pass allow all pixels, second pass allow only low sigma pixels. */
for (siglim=9995.; siglim>0.; siglim=siglim-9990.) {
  for (nso=3; nso<=7; ++nso) {
    if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
    for (ii=1; ii<(ecol-1); ++ii) {
      edge = nsx_find_real_image_row( 1, ii, nso );
      row1 = cnint(edge) - 3;
      row2 = cnint(edge) + SPX[nso].numpro + 2;
      for (jj=(row1+4); jj<=(row2-4); ++jj) {
        c1 = ii-1; c2 = ii+1;
        r1 = jj-4; r2 = jj+4;
        nsx_sigma_subsection(nc,nr,IMG[0].clnimg,sigimg,valimg,c1,c2,r1,r2,ii,jj,siglim);
      }
    }
  }
}

/* Flag strong CRs and affected adjacent pixels. */
maxsig=12.0; adjsig=4.0;
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  col1=0;
  col2=ecol-1;
  for (ii=col1; ii<=col2; ++ii) {
    edge = nsx_find_real_image_row( 1, ii, nso );
    row1 = cnint(edge) - 3;
    row2 = cnint(edge) + SPX[nso].numpro + 2;
    for (jj=row1; jj<=row2; ++jj) {
      pixno = ii + (jj * nc);
      if (flgimg[pixno] < 0.1) {
        if (valimg[pixno] >  0.    ) {
        if (sigimg[pixno] >= maxsig) {
          flgimg[pixno]=2.;
          c1 = ii-1;   if (c1 < col1) c1=col1;
          c2 = ii+1;   if (c2 > col2) c2=col2;
          r1 = jj-1;   if (r1 < row1) r1=row1;
          r2 = jj+1;   if (r2 > row2) r2=row2;
          for (iii=c1; iii<=c2; ++iii) {
          for (jjj=r1; jjj<=r2; ++jjj) {
            ppp = iii + (jjj * nc);
            if ((ppp != pixno)&&(valimg[ppp] > 0.)) {
              if ((flgimg[ppp] < 0.1)&&(sigimg[ppp] > adjsig)) { flgimg[ppp]=1.; }
            }
          }}
        }}
      }
    }
  }
}

/* Replace found bad pixels with local median. */
flgcount=0;
for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) {
  pixno = ii + (nc * jj);
  if (flgimg[pixno] > 0.9) {
    IMG[0].clnimg[pixno] = nsx_median_box(IMG[0].image,flgimg,nc,nr,ii,jj,1,4,arr,maxarr);
    ++flgcount;
  }
}}
printf("  %7d other pixels corrected...\n",flgcount);


/* #@# */
nsx_write_general_image( "image.fits" , IMG[0].image,  nc, nr );
nsx_write_general_image( "clnimg.fits", IMG[0].clnimg, nc, nr );
nsx_write_general_image( "flgimg.fits",        flgimg, nc, nr );
nsx_write_general_image( "valimg.fits",        valimg, nc, nr );
nsx_write_general_image( "sigimg.fits",        sigimg, nc, nr );
/* #@# */

free(flgimg); free(valimg); free(sigimg); 
return;
}




/* ----------------------------------------------------------------------
 clean sub image.
*/
int nsx_subimage_clean( float image[], int nc, int nr, float rejimg[] )
{
/**/
int nrej,ppp,iii,jjj,ii1,ii2,jj1,jj2,ii,jj,pixno;
/**/
float sigimg[100+(nc*nr)];
float image2[100+(nc*nr)];
/**/
const float siglim = 4.;
const float adjsiglim = 2.;
const int isigrad = 2;   /* better for sky lines */
const int jsigrad = 4;
const int imedrad = 1;   /* better for sky lines */
const int jmedrad = 4;
/**/

/* Copy. Clear. Sigs. */
for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) { 
  pixno = ii + (jj * nc); 
  image2[pixno] = image[pixno]; 
  sigimg[pixno] = nsx_rmssigdev( image, nc, nr, isigrad, jsigrad, ii, jj );
}}

/* Replace high sig pixels. */
nrej=0;
for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) {
  pixno = ii + (jj * nc);
  if (sigimg[pixno] > siglim) {
    image2[pixno] = nsx_submedian( image, nc, nr, imedrad, jmedrad, ii, jj );
    rejimg[pixno] = 1.;
    ++nrej;
  }
}}

/* Adjacents. */
for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) {
  pixno = ii + (jj * nc);
  if (rejimg[pixno] > 0.) {
    ii1=ii-1; ii2=ii+1; jj1=jj-1; jj2=jj+1;
    if (ii1 < 0   ) ii1=0;
    if (ii2 > nc-1) ii2=nc-1;
    if (jj1 < 0   ) jj1=0;
    if (jj2 > nr-1) jj2=nr-1;
    for (iii=ii1; iii<=ii2; ++iii) {
    for (jjj=jj1; jjj<=jj2; ++jjj) {
      ppp = iii + (jjj * nc);
      if (rejimg[ppp] == 0.) {
        if (sigimg[ppp] > adjsiglim) {
          image2[ppp] = nsx_submedian( image, nc, nr, imedrad, jmedrad, iii, jjj );
          rejimg[ppp] = 2.;
           ++nrej;
        }
      }
    }}
  }
}}

/*
nsx_write_general_image( "testsig.fits", sigimg, nc, nr );
nsx_write_general_image( "testrej.fits", rejimg, nc, nr );
nsx_write_general_image( "testmed.fits", image2, nc, nr );
cpauseit();
*/

/* Copy back. */
for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) { pixno = ii + (jj * nc); image[pixno] = image2[pixno]; }}

return(nrej);
}
        


/* ----------------------------------------------------------------------
 Compute highest weighted deviation and index number.
*/ 
void nsx_hidev( int nn, double xx[], double yy[], double ww[], int order, double xoff, double coef[], 
                double *hidev, int *hiii )
{
/**/
int ii;
double dev;
/**/
*hiii=-1; *hidev=-1.;
for (ii=0; ii<nn; ++ii) { if (ww[ii] > 0.) {
  dev = ABS(( yy[ii] - cpolyval(order+1,coef,(xx[ii]-xoff)) ));
  if (dev > *hidev) { *hidev=dev; *hiii=ii; }
}}
return;
}


/* ----------------------------------------------------------------------
 Compute highest weighted deviation and index number.
 Also return:
     'yrms'   (RMS of Y values from median of Y values (ww>0), excluding 20% worse values ).
     'devmed' (median of deviations from fit (ww>0) ).
     'devrms' (RMS of deviations from fit (ww>0), excluding 20% worse values ).
     'hidev'  (high deviation from fit (ww>0) ).
*/ 
void nsx_hidevrms( int nn, double xx[], double yy[], double ww[], int order, double xoff, double coef[], 
                   double *yrms, double *devmed, double *devrms, double *hidev, int *hiii )
{
/**/
int ii,narr,narr80;
double dev,ff,devarr[3000],arr[3000],ymedian,sum;
/**/

/* Y values: Median and rms. */
narr=0;
for (ii=0; ii<nn; ++ii) { if (ww[ii] > 0.) {
  arr[narr] = yy[ii];
  ++narr;
}}
ymedian = cfind_median8(narr,arr);

/* Y values: Compute RMS excluding the 20% largest deviations. */
for (ii=0; ii<narr; ++ii) { devarr[ii] = ABS((arr[ii] - ymedian)); }
cqcksrt8(narr,devarr);
narr80 = cnint( 0.8 * (double)narr );
sum = 0.;
for (ii=0; ii<narr80; ++ii) { sum = sum + (devarr[ii] * devarr[ii]); }
*yrms = sqrt(( sum / (double)narr80 ));

/* Fit deviations: compute RMS relative to fit excluding 20% largest deviations. */
narr=0;
for (ii=0; ii<nn; ++ii) { if (ww[ii] > 0.) {
  ff = cpolyval(order+1,coef,(xx[ii]-xoff));
  devarr[narr] = ABS(( yy[ii] - ff ));
  ++narr;
}}
*devmed = cfind_median8(narr,devarr);
narr80 = cnint( 0.8 * (double)narr );
sum = 0.;
for (ii=0; ii<narr80; ++ii) { sum = sum + (devarr[ii] * devarr[ii]); }
*devrms = sqrt(( sum / (double)narr80 ));

/* Highest deviation. */
*hidev=-1.; *hiii=-1; 
for (ii=0; ii<nn; ++ii) { if (ww[ii] > 0.) {
  ff = cpolyval(order+1,coef,(xx[ii]-xoff));
  dev = ABS(( yy[ii] - ff ));
  if (dev > *hidev) { *hidev=dev; *hiii=ii; }
}}

return;
}


/* ----------------------------------------------------------------------
  Fit a polynomial with rejections (maxsig usually 5.0)  (maxiter ~ nn/5)
*/
void nsx_fitpoly_reject( int nn, double xx[], double yy[], double ww[], int order, int maxiter, 
                         double maxsig, double *xoff, double coef[], int echo )
{
/**/
int nrej,ii,reject,iter,hiii;
double ff,xv,yrms,devmed,devrms,hidev,sigs,devsigs,hisigs;
FILE *outfu;
/**/
/* Fit. */
reject= 1;
iter  = 1;
yrms=0.; devmed=0.; devrms=0.; hidev=0.; hiii=-1; sigs=0.; devsigs=0.; hisigs=0.;
*xoff=0.; coef[0]=0.; coef[1]=0.; coef[2]=0.; coef[3]=0.; coef[4]=0.; coef[5]=0.;
nrej=0;
while ((reject)&&(iter < maxiter)) {
  if (GJ_polyfit(nn,xx,yy,ww,order,0,xoff,coef) != 1) { printf("***error:nfr: fit failed.\n"); exit(1); }
  nsx_hidevrms( nn, xx, yy, ww, order, *xoff, coef, &yrms, &devmed, &devrms, &hidev, &hiii );
  if ((yrms > 0.)&&(devrms > 0.)&&(hiii >= 0)) {
    sigs   = hidev / yrms;
    devsigs= devmed / devrms;
    hisigs = hidev  / devrms;
  } else { sigs=0.; devsigs=0.; hisigs=0.; }
  if ((sigs > maxsig)||(hisigs > maxsig)) { 
      ww[hiii]=0.; reject=1; 
      ++nrej;
      if (echo) { 
        printf("Reject: iter=%2d  xx=%f  sigs=%f  hisigs=%f  devrms=%f  devsigs=%f  hidev=%f\n",
           iter,xx[hiii],sigs,hisigs,devrms,devsigs,hidev); 
      }
  } else { reject=0; }
  ++iter;
}
printf("nsx_fitpoly_reject: nn=%3d  maxiter=%3d  maxsig=%5.2f  order=%2d  nrej=%3d\n",nn,maxiter,maxsig,order,nrej);
if (echo) {
  printf(" Final: iter=%2d  xx=%f  sigs=%f  hisigs=%f  devrms=%f  devsigs=%f  hidev=%f\n",
           iter,xx[hiii],sigs,hisigs,devrms,devsigs,hidev); 
  outfu = fopen_write("fit.dat");  printf("writing 'fit.dat'..\n");
  for (ii=0; ii<nn; ++ii) {
    xv = xx[ii] - (*xoff);
    ff = cpolyval(order+1, coef, xv );
    fprintf(outfu,"%20.12e %20.12e %20.12e %20.12e \n",xx[ii],yy[ii],ff,yy[ii]-ff);
  }
  fclose(outfu);
}
return;
}


/* ----------------------------------------------------------------------
  Find peak pixel near centroid.  Returns array element index number.
*/
int nsx_peak( int nn, double xx[], double yy[], double cent, double radius )
{
/**/
int ii,kk;
double peak;
/**/
kk=0; peak=-999999.;
for (ii=0; ii<nn; ++ii) {
  if (ABS((xx[ii] - cent)) <= radius) {
    if (yy[ii] > peak) { peak=yy[ii]; kk=ii; }
  }
}
return(kk);
}


/* ----------------------------------------------------------------------
  Find (2D) full width of a sky line.
*/
double nsx_fwhm_skyline( int icol, int colrad, int irow, int rowrad, int nc, int nr, float image[] )
{
/**/
int nn,pixno,ii,jj;
double flux,sum,bck,area,peak,fwhm,xx[100],yy[100];
/**/

nn=0;
for (ii=icol-colrad; ii<=icol+colrad; ++ii) {
  sum=0.;
  for (jj=irow-rowrad; jj<=irow+rowrad; ++jj) {
    pixno = ii + (jj * nc);
    sum = sum + image[pixno];
  }
  xx[nn] = (double)ii;
  yy[nn] = sum;
  ++nn;
}

/* FWHM. */
bck=yy[0]; for (ii=1; ii<nn; ++ii) { if (yy[ii] < bck) bck=yy[ii]; }
area=0.; peak=0.;
for (ii=0; ii<nn; ++ii) { 
  flux = yy[ii] - bck;
  area = area + flux;
  if (flux > peak) peak=flux;
}
if (peak > 0.) { fwhm = 0.939437 * area / peak; } else { fwhm=0.; }


/*
sprintf(wrd,"fw%4.4d.tbl",irow);
outfu = fopen_write(wrd);
for (ii=0; ii<nn; ++ii) { fprintf(outfu,"%f %f\n",xx[ii],yy[ii]); }
fclose(outfu);
*/

return(fwhm);
}


/* ----------------------------------------------------------------------
  Fit a straight line to a set of points.
  Input: nn    : number of points.
         xx[]  : x points.
         yy[]  : y points.
         ww[]  : weight points.
 Output: coef  : Coeffecient array (only coef[0] and coef[1]).
*/
/*@@*/
void fit_straight_line( int nn, double xx[], double yy[], double ww[], double coef[] )
{
/**/
double c11,c12,c13,c22,c23;
/**/
int ii;
/**/
/* Compute constants for fitting a straight line. */
c11 = 0.;
c12 = 0.;
c13 = 0.;
c22 = 0.;
c23 = 0.;
for (ii=0; ii<nn; ++ii) {
  if (ww[ii] > 0.) {
    c11 = c11 + ww[ii];
    c12 = c12 + xx[ii] * ww[ii];
    c13 = c13 + yy[ii] * ww[ii];
    c22 = c22 + xx[ii] * xx[ii] * ww[ii];
    c23 = c23 + yy[ii] * ww[ii] * xx[ii];
  }
}
/* Solve for coef[0] and coef[1]. */
coef[1] = ( (c13 * c12) - (c23 * c11) ) / ( (c12 * c12) - (c22 * c11) );
coef[0] = ( c13 - (c12 * coef[1]) ) / c11;
return;
}



/* ----------------------------------------------------------------------
   For more speed try: math_convolve_gaussian().
   Convolve a gaussian into an array of points.
   That is, smooth a spectrum with a gaussian of width 'fwhm'.
     Input: nn[]  : number of points in original array.
            xx[]  : x values array.
            aa[]  : Original array (or spectrum).
            fwhm  : FWHM of convolving gaussian.
            cutoff: How far out to go on gaussian in terms of FWHMs (e.g. 4.0).
    Output: aac[] : aa[] convolved or smoothed with gaussian.
*/
/*@@*/
void math_convolve_gaussian0( int nn, double xx[], double aa[], double fwhm, double fw[], double cutoff, double aac[] )
{
/**/
const double ln2 = 0.693147181;      /* natural logarithm of 2. */
const double c = 0.469718639;        /*  c = sqrt(ln2/pi)       */
/**/
int ii,jj;
double xv,rr,sum,wsum,hwhm,hwhm2,lim,neglim;
/**/
hwhm  = fwhm / 2.;
hwhm2 = hwhm * hwhm;
lim   = cutoff * hwhm;
neglim= -1. * lim;
for (ii=0; ii<nn; ++ii) {
if (fw[ii] > 0.) {
  if (ii%1000 == 0) printf("%7d of %7d\n",ii,nn);
  sum = 0.;
  wsum= 0.;
  for (jj=0; jj<nn; ++jj) {
    xv = xx[jj] - xx[ii];
    if ((xv > neglim)&&(xv < lim)) {
      xv = -1. * (xv * xv / hwhm2) * ln2;
      rr = c * exp((xv));
      sum = sum + ( aa[jj] * rr );
      wsum= wsum+ rr;
    }
  }
  aac[ii] = sum / wsum;
}
}
return;
}


/* ----------------------------------------------------------------------
   For more speed try: math_convolve_gaussian().
   Convolve a gaussian into an array of points.
   That is, smooth a spectrum with a gaussian of width 'fwhm'.
     Input: nn[]  : number of points in original array.
            xx[]  : x values array.
            aa[]  : Original array (or spectrum).
            fw[]  : FWHMs at each array element point for convolving gaussian.
                    (only compute smoothed pixel if fw[] > 0, otherwise aac=0)
            cutoff: How far out to go on gaussian in terms of FWHMs (e.g. 4.0).
    Output: aac[] : aa[] convolved or smoothed with gaussian.
*/
/*@@*/
void math_convolve_gaussian_fwhm( int nn, double xx[], double aa[], double fw[], double cutoff, double aac[] )
{
/**/
const double ln2 = 0.693147181;      /* natural logarithm of 2. */
const double c = 0.469718639;        /*  c = sqrt(ln2/pi)       */
/**/
int ii,jj;
double xv,rr,sum,wsum,hwhm,hwhm2,lim,neglim;
/**/
for (ii=0; ii<nn; ++ii) {
  if (fw[ii] > 0.) {
    hwhm  = fw[ii] / 2.;
    hwhm2 = hwhm * hwhm;
    lim   = cutoff * hwhm;
    neglim= -1. * lim;
    sum = 0.;
    wsum= 0.;
    for (jj=0; jj<nn; ++jj) {
      xv = xx[jj] - xx[ii];
      if ((xv > neglim)&&(xv < lim)) {
        xv = -1. * (xv * xv / hwhm2) * ln2;
        rr = c * exp((xv));
        sum = sum + ( aa[jj] * rr );
        wsum= wsum+ rr;
      }
    }
    aac[ii] = sum / wsum;
  } else {
    aac[ii] = 0.;
  }
}
return;
}



/* ----------------------------------------------------------------------
  Find peak in cross correlation..
*/
double nsx_find_cc_peak( int ncc, double xcc[], double ycc[] )
{
/**/
int ii,ii1,ii2,jj,hiii;
double num,hidif,bck,dif;
/**/
/* Find best peak after subtracting off background. */
hidif=0.; hiii=-1;
for (ii=8; ii<ncc-8; ++ii) {
  bck=0.; num=0.;
  ii1=ii-7; ii2=ii-4; for (jj=ii1; jj<=ii2; ++jj) { bck=bck+ycc[jj]; num=num+1.; }
  ii1=ii+4; ii2=ii+7; for (jj=ii1; jj<=ii2; ++jj) { bck=bck+ycc[jj]; num=num+1.; }
  bck = bck / num;
  dif = ycc[ii] - bck; 
  if (dif > hidif) { hidif=dif; hiii=ii; }
}
if (hiii < 0) { printf("***error: finding sky shift.\n"); exit(1); }
return(xcc[hiii]);
}


/* ----------------------------------------------------------------------
  Extract sky spectrum and compute skyline shift in pixels.
  Used in nsx_extract_spectrum() routine.
  IMGSKY : Sky data image (A image for both A only and A-B data).
  Loads .spsky[], .spwav[], and .spdsp[].  The 'spwav' values are shifted
  by a median pixel value (written to log file).
*/
void nsx_compute_skyline_shift( IMGtype IMGSKY, SPXtype SPX[], char nsxdir[], char nsxout[] )
{
/**/
int ecol,nso,niter,icol;
int kk,ii,nn,jjoff;
int ncc,iss,ccshift,narr,nss,ssnso[3000];
int skynum[9];
/**/
double disp,rcol,xv,sum,npx,rblo,rbhi,rb1,rb2,skyshift;
double fwhm_radius,radius,cent,xx[3000],yy[3000];
double fullratio,mratio,ratio,mrms,rms,fwhm_median,median,arr[1000],peak,fwhm;
double skywave[9][100],skywgt[9][100],sscent[1000],sscol[1000];
double ssresi[1000],sspeak[1000],ssfwhm[1000],sswave[1000];
double xcc[3000],ycc[3000],skytp[3000];
/**/
char wrd[200];
char line[200];
/**/
FILE *infu;
FILE *outfu;
/**/
/*
double wave1,wave2;
FILE *outfu2;
int nn8;
double fwhm,xx8[100],yy8[100],ww8[100],coef8[9];
*/
/**/

printf("Compute skyline shift.\n");

/* For each order, load wavelength scale and dispersion. */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IMGSKY.nc/2; } else { ecol=IMGSKY.nc; }
  for (icol=0; icol<ecol; ++icol) {
    xv  = (double)icol - WSC[nso].xoff;
    SPX[nso].spwav[icol] = cpolyval(WSC[nso].order+1,WSC[nso].coef,xv);
  }
/* Compute dispersion (angstroms per pixel) (about -2.8 ang/pix). */
  for (icol=0; icol<ecol; ++icol) {
    if (icol ==    0  ) { SPX[nso].spdsp[icol] = SPX[nso].spwav[1]      - SPX[nso].spwav[0];          } else {
    if (icol == ecol-1) { SPX[nso].spdsp[icol] = SPX[nso].spwav[ecol-1] - SPX[nso].spwav[ecol-2];     } else {
                        { SPX[nso].spdsp[icol] =(SPX[nso].spwav[icol+1] - SPX[nso].spwav[icol-1])/2.; }      }}
  }
}

/* Extract sky spectrum from corrected background image for each order (in object window). */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IMGSKY.nc/2; } else { ecol=IMGSKY.nc; }
  jjoff = ((7-nso) * 200);
  rblo = (double)jjoff;
  rbhi = (double)(jjoff + SPX[nso].numpro);
  for (icol=0; icol<ecol; ++icol) {
    rb1 = (double)jjoff + (SPX[0].asp1[0] / ARCSEC_PER_PIXEL);
    if (rb1 < rblo) rb1 = rblo;
    if (rb1 > rbhi) rb1 = rbhi;
    rb2 = (double)jjoff + (SPX[0].asp2[0] / ARCSEC_PER_PIXEL);
    if (rb2 < rblo) rb2 = rblo;
    if (rb2 > rbhi) rb2 = rbhi;
    sum = nsx_fractional_pixel_rb( IMGSKY.nc, IMGSKY.bckimg, rb1, rb2, icol );
    npx = ABS((rb2 - rb1));
    if (npx > 0.) { SPX[nso].spsky[icol] = sum/npx; } else { SPX[nso].spsky[icol] = 0.; }
    SPX[nso].spsky[icol] = SPX[nso].spsky[icol] / IMGSKY.exptime;
  }
}


/* Read in sky line wavelength data. */
for (nso=3; nso<=7; ++nso) {
  sprintf(wrd,"%scal/skylines_nsx%d.tbl",nsxdir,nso);
  infu = fopen_read(wrd);
  kk=0;
  while (fgetline(line,infu)) { if (line[0] != '|') {
    skywave[nso][kk]= GLV(line,1);
    skywgt[nso][kk] = GLV(line,2);
    ++kk;
    if (kk > 90) { printf("***error: too many sky lines in nso=%d\n",nso); exit(1); }
  }}
  fclose(infu);
  skynum[nso]=kk;
}



/* Cross correlation .. */
narr=0;
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IMGSKY.nc/2; } else { ecol=IMGSKY.nc; }
  for (icol=0; icol<3000; ++icol) { skytp[icol]=0.; }
  sprintf(wrd,"%s/cal/sky_template%d.tbl",nsxdir,nso);
  infu = fopen_read(wrd);
  while (fgetline(line,infu)) { if (line[0] != '|') {
    icol = GLV(line,1);
    skytp[icol] = GLV(line,2);
  }}
  fclose(infu);
  ncc=0;
  for (iss=-200; iss<200; ++iss) {
    sum=0.;
    for (icol=0; icol<ecol; ++icol) {
      ii = icol + iss;
      if ((ii >= 0)&&(ii < ecol)) { sum = sum + (skytp[icol] * SPX[nso].spsky[ii]); }
    }
    xcc[ncc] = iss;
    ycc[ncc] = sum;
    ++ncc;
  }

/* #@#
  sprintf(wrd,"cc%d.dat",nso);
  outfu = fopen_write(wrd);
  for (ii=0; ii<ncc; ++ii) { fprintf(outfu," %15.7e %15.7e \n",xcc[ii],ycc[ii]); }
  fclose(outfu);
   #@# */

  arr[narr] = nsx_find_cc_peak( ncc, xcc, ycc );

/* #@#
  sprintf(wrd,"jsky%d.tbl",nso);
  outfu = fopen_write(wrd);
  fprintf(outfu,"|icol | skytp         | spsky         | sss           |\n");
  for (icol=0; icol<ecol; ++icol) {
    spsky_icol = icol + arr[narr];
    fprintf(outfu," %5d %15.7e %15.7e %15.7e \n",icol,skytp[icol],SPX[nso].spsky[icol],SPX[nso].spsky[spsky_icol]);
  }
  fclose(outfu);
#@# */

  ++narr;
}
ccshift = cnint(( cfind_median8(narr,arr) ));
printf("Use skyline whole pixel shift of = %d .\n",ccshift);
fprintf(logfu,"Use skyline whole pixel shift of = %d .\n",ccshift);


/* Skyline shift. */
nss=0;
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IMGSKY.nc/2; } else { ecol=IMGSKY.nc; }
/* Copy sky spectrum on pixel scale and apply cross correlation peak shift. */
  nn=0;
  for (icol=0; icol<ecol; ++icol) {
    ii = icol + ccshift;
    if ((ii >= 0)&&(ii < ecol)) {
      yy[nn] = SPX[nso].spsky[ii];
      xx[nn] = (double)icol;
      ++nn;
    }
  }
/* Centroid sky lines for Sky Shift. */
  radius=3.; niter=4;  fwhm_radius=4.;
  for (ii=0; ii<skynum[nso]; ++ii) { if (skywgt[nso][ii] > 0.) {
    xv   = skywave[nso][ii] - WSC[nso].xoffinv;
    rcol = cpolyval(WSC[nso].orderinv+1,WSC[nso].coefinv,xv);
    cent = nsx_centroid2( nn, xx, yy, rcol, radius, niter, 1 );
    fwhm = nsx_fwhm_1D( nn, xx, yy, cent, fwhm_radius, &peak, 1 );
    if (cent > -1.e+20) {
      ssnso[nss] = nso;
      sscol[nss] = rcol;
      sscent[nss]= cent;
      ssresi[nss]= cent-rcol;
      sswave[nss]= skywave[nso][ii];
      ssfwhm[nss]= fwhm;
      sspeak[nss]= peak;
      ++nss;
    }
  }}
}

/* FWHM median from all orders. */
narr=0;
for (ii=0; ii<nss; ++ii) { arr[narr]=ssfwhm[ii]; ++narr; }
fwhm_median = cfind_median8(narr,arr);

/* Medians for different orders. */
for (nso=3; nso<=7; ++nso) {
  narr=0;
  for (ii=0; ii<nss; ++ii) {
    if (ssnso[ii] == nso) { arr[narr]=ssresi[ii]; ++narr; }
  }
  median = cfind_median8(narr,arr);
  printf("Echelle order %d: sky line shift in pixels (median)=%9.5f\n",nso,median);
  fprintf(logfu,"Echelle order %d: sky line shift in pixels (median)=%9.5f\n",nso,median);
}

/* Median from all orders. */
narr=0;
for (ii=0; ii<nss; ++ii) { arr[narr]=ssresi[ii]; ++narr; }
median = cfind_median8(narr,arr);

/* Write skyline residuals to file. */
sprintf(wrd,"%s%s-skyres.tbl",nsxout,IMGSKY.root);
outfu = fopen_write(wrd); 
fprintf(outfu,"|ord| column |centroid|resid_pix|resid_ang|wavelength|shift_res|fwhm_pix| peak   |fwhm_ang|\n");
rms=0.; mrms=0.; ratio=0.; mratio=0.;
for (ii=0; ii<nss; ++ii) {
  nso = ssnso[ii];
  icol= cnint(( sscent[ii] ));
  disp= SPX[nso].spdsp[icol];
  fprintf(outfu," %3d %8.3f %8.3f %9.4f %9.4f %10.3f %9.4f %8.3f %8.1f %8.3f \n",
       ssnso[ii], sscol[ii], sscent[ii], ssresi[ii], ssresi[ii]*disp, sswave[ii], ssresi[ii]-median,
       ssfwhm[ii], sspeak[ii], ssfwhm[ii]*ABS((disp)) );
  rms = rms + ((ssresi[ii]-median)*(ssresi[ii]-median));
}
fclose(outfu);
if (nss > 1) { 
  rms = sqrt(( rms / (double)nss )); 
  if (rms > 0.) { ratio = ABS((median / rms)); }
  mrms= rms / sqrt((double)nss);
  if (mrms > 0.) { mratio = ABS((median / mrms)); }
  skyshift = (double)ccshift + median;
  fullratio= ABS(( skyshift / mrms ));
  printf(      " Skyline shift (all orders): %9.5f(pixels)  rms=%9.5f [%6.2f]  mrms=%9.5f [%6.2f] {%6.1f} num=%d  fwhm=%8.3f\n",skyshift,rms,ratio,mrms,mratio,fullratio,nss,fwhm_median);
  fprintf(logfu," Skyline shift (all orders): %9.5f(pixels)  rms=%9.5f [%6.2f]  mrms=%9.5f [%6.2f] {%6.1f} num=%d  fwhm=%8.3f\n",skyshift,rms,ratio,mrms,mratio,fullratio,nss,fwhm_median);
}



/* Use no skyline shift if constructing atm.abs. files. */
if (CONSTRUCT_ATMABS) {
  printf("WARNING: use skyline shift median of zero. \n"); 
  median = 0.;
}



/* Apply sky shift if large enough.  Apply pixel shift times dispersion. */
if ((nss > 1)&&(ratio > 0.)&&(rms > 0.)) {
  if (fullratio > 5.) {
    skyshift = (double)ccshift + median;
    printf("Apply sky line shift of %9.5f pixels to all echelle orders.\n",skyshift); 
    fprintf(logfu,"Apply sky line shift of %9.5f pixels to all echelle orders.\n",skyshift); 
    for (nso=3; nso<=7; ++nso) {
      if (nso == 7) { ecol=IMGSKY.nc/2; } else { ecol=IMGSKY.nc; }
      for (icol=0; icol<ecol; ++icol) {
        SPX[nso].spwav[icol] = SPX[nso].spwav[icol] - (SPX[nso].spdsp[icol] * skyshift);
      }
    }
  } else {
    printf(       "No sky line shift applied, not significant enough.\n");
    fprintf(logfu,"No sky line shift applied, not significant enough.\n");
  }
}


/* #@# temp write
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IMGSKY.nc/2; } else { ecol=IMGSKY.nc; }

  sprintf(wrd,"junkcent%d.tbl",nso);
  outfu2 = fopen_write(wrd);
  fprintf(outfu2,"|nso| wave    | cent    | diff  | rcol   | difpx  |\n");

  sprintf(wrd,"junksky%d.dat",nso);
  outfu = fopen_write(wrd);

  nn=0;
  for (icol=ecol-1; icol>=0; --icol) {
    xx[nn] = SPX[nso].spwav[icol];
    yy[nn] = SPX[nso].spsky[icol];
    fprintf(outfu,"%f %f\n",xx[nn],yy[nn]);
    ++nn;
  }
  fclose(outfu);
  printf("wrote '%s'\n",wrd);

  sprintf(wrd,"junksky%d.draw",nso);
  outfu = fopen_write(wrd);
  fprintf(outfu,"sci 3 ; sls 1\n");
  for (kk=0; kk<skynum[nso]; ++kk) {
    fprintf(outfu,"%f 0\n",skywave[nso][kk]);
    fprintf(outfu,"%f %f\n",skywave[nso][kk],skywgt[nso][kk]);
    fprintf(outfu,"draw\n");
    cent = nsx_centroid2(nn,xx,yy,skywave[nso][kk],14.,4,1);
    printf("nso=%d  skywave=%f  cent=%f\n",nso,skywave[nso][kk],cent);
    if (cent > 0.) {

      xv   = skywave[nso][kk] - WSC[nso].xoffinv;
      rcol = cpolyval(WSC[nso].orderinv+1,WSC[nso].coefinv,xv);

      xv   = (rcol - 0.5) - WSC[nso].xoff;
      wave1= cpolyval(WSC[nso].order+1,WSC[nso].coef,xv);

      xv   = (rcol + 0.5) - WSC[nso].xoff;
      wave2= cpolyval(WSC[nso].order+1,WSC[nso].coef,xv);
      disp = wave1 - wave2;

      fprintf(outfu2," %3d %9.2f %9.2f %7.3f %8.2f %8.4f \n",
          nso, skywave[nso][kk], cent, cent-skywave[nso][kk], rcol, (cent-skywave[nso][kk])/disp );
    }

  }
  fclose(outfu);
  fclose(outfu2);
}
    #@# */


return;
}



/* ----------------------------------------------------------------------
   Fit a line and then do a relative rms. 
   Returns rms value.
*/
double nsx_straight_rms( int nn, double xx[], double yy[], double ww[], double oo[], double aa[], char datfile[] )
{
/**/
double mean,ff,xv,xoff,rms,coef[9];
int order,ii;
/**/
FILE *outfu;
/**/

order=1; xoff=0.;
fit_straight_line( nn, xx, yy, ww, coef );
outfu = fopen_write(datfile);
mean=0.;
for (ii=0; ii<nn; ++ii) { mean = mean + yy[ii]; }
mean=mean/(double)nn;
rms = 0.;
for (ii=0; ii<nn; ++ii) {
  xv = xx[ii] - xoff;
  ff = cpolyval(order+1,coef,xv);
  rms = rms + ( (yy[ii] - ff) * (yy[ii] - ff) );
  fprintf(outfu,"%f %f %f %f %f\n",xx[ii],yy[ii],ff,oo[ii],aa[ii]*mean); 
}
fclose(outfu);
rms = sqrt(( rms / (double)nn ));
return(rms);
}


/* ----------------------------------------------------------------------
   Decide on best atmospheric transmission data set. 
   Apply correction.
*/
void nsx_find_best_mktfile( IMGtype IMG, SPXtype SPX[], MKTtype MKT[], double mktwave[], char nsxdir[] )
{
/**/
int vv,ivv,aa,iaa,icol,seg,ecol,ii,nso;
/**/
int nc = IMG.nc;
/**/
char wrd[200];
/**/
FILE *binfu;
/**/
double wave,atmabs;
/**/
double lotot,tot,mxx[800],myy[800],moo[800],maa[800],mww[800];
int mnn,loaa,lovv;
double minwv[9],maxwv[9];
/**/
MKTtype *MKTA[41][11];  /* All files. All vapors and airmasses. */
/**/
FILE *outfu;


printf("Find best MKT data set...\n");

/* Allocate and clear. */
/* Airmass is 1.0 to 2.0 by 0.1 .. */
/* Water vapor goes from 1.0 to 5.0 by 0.1 .. */
for (aa=10; aa<=20; aa=aa+1) { iaa = aa - 10;
  for (vv=10; vv<=50; vv=vv+1) { ivv = vv - 10;
    MKTA[ivv][iaa] = (MKTtype *)calloc((NUMMKT),sizeof(MKTtype));
    for (ii=0; ii<NUMMKT; ++ii) {
      MKTA[ivv][iaa][ii].tran[0]=0.;
      MKTA[ivv][iaa][ii].tran[1]=0.;
      MKTA[ivv][iaa][ii].tran[2]=0.;
      MKTA[ivv][iaa][ii].tran[3]=0.;
      MKTA[ivv][iaa][ii].tran[4]=0.;
    }
  }
}
for (ii=0; ii<NUMMKT; ++ii) { mktwave[ii]=0.; }


/* Read wavelengths from binary. */
sprintf(wrd,"%scal/mkts_W.bin",nsxdir); printf("Read '%s'.\n",wrd);
binfu = fopen_read(wrd);
for (ii=0; ii<NUMMKT; ++ii) { fread((char *)&mktwave[ii], 1, sizeof(double), binfu ); }
fclose(binfu);

/* #@# ONLY CONSIDER nso=3 for now.. */
/* #@# ONLY CONSIDER nso=3 for now.. */
/* #@# ONLY CONSIDER nso=3 for now.. */
/* #@# ONLY CONSIDER nso=3 for now.. */
/* #@# ONLY CONSIDER nso=3 for now.. */

/* Read all transmissions, all orders, vapors, and airmasses. */
for (nso=3; nso<=7; ++nso) {
if (nso == 3) {
  sprintf(wrd,"%scal/mkts_%d.bin",nsxdir,nso); printf("Read '%s'.\n",wrd);
  binfu = fopen_read(wrd);
  for (aa=10; aa<=20; aa=aa+1) { iaa = aa - 10;
    for (vv=10; vv<=50; vv=vv+1) { ivv = vv - 10;
      for (ii=0; ii<NUMMKT; ++ii) {
        fread((char *)&MKTA[ivv][iaa][ii].tran[nso-3], 1, sizeof(double), binfu );
      }
    }
  }
  fclose(binfu);
}
}

/* Segments. */
lotot=9.e+20; loaa=0; lovv=0;
minwv[0]=19700.; maxwv[0]=19940.;
minwv[1]=19940.; maxwv[1]=20700.;
for (seg=0; seg<2; ++seg) {

  sprintf(wrd,"seg%d.dat",seg);
  outfu = fopen_write(wrd);
/* Check atmospheric aborption. */
  for (nso=3; nso<=7; ++nso) {
  if (nso == 3) {
    if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
    lotot=9.e+20; loaa=0; lovv=0;
    for (aa=10; aa<=20; aa=aa+1) { iaa = aa - 10;
    for (vv=10; vv<=50; ++vv) { ivv = vv - 10;
      mnn=0;
      for (icol=0; icol<ecol; ++icol) {
        wave = SPX[nso].spwav[icol];
        if ((wave > minwv[seg])&&(wave < maxwv[seg])) {
          ii      = cneari_bs( wave, NUMMKT, mktwave );
          atmabs  = MKTA[ivv][iaa][ii].tran[nso-3];   if (atmabs < 1.e-10) atmabs=1.e-10;
          mxx[mnn]= wave;
          myy[mnn]= SPX[nso].spobj[icol] / atmabs;
          mww[mnn]= 1.0;
          moo[mnn]= SPX[nso].spobj[icol];
          maa[mnn]= atmabs;
          mnn = mnn + 1;
        }
      }
      sprintf(wrd,"junk_v%d_a%d_seg%d.dat",vv,aa,seg);
      tot = nsx_straight_rms( mnn, mxx, myy, mww, moo, maa, wrd );
      if (tot < lotot) { lotot=tot; loaa=aa; lovv=vv; }
      fprintf(outfu,"%2d %2d %9.4f\n",vv,aa,tot);
    }
    }
    printf("NOTE: best atmospheric match for seg %2d : lotot=%f  loaa=%d   lovv=%d\n",seg,lotot,loaa,lovv);
  }
  }
  fclose(outfu);

}

/* TEMPORARY: use the last segment.. */
/* TEMPORARY: use the last segment.. */
/* TEMPORARY: use the last segment.. */

/* Load up atmospheric transmission. */
ivv = lovv - 10;
iaa = loaa - 10;
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (icol=0; icol<ecol; ++icol) {
    ii = cneari_bs( SPX[nso].spwav[icol], NUMMKT, mktwave );
    SPX[nso].spatm[icol] = MKTA[ivv][iaa][ii].tran[nso-3];
    SPX[nso].spoac[icol] = SPX[nso].spobj[icol] / SPX[nso].spatm[icol];
    SPX[nso].speac[icol] = SPX[nso].sperr[icol] / SPX[nso].spatm[icol];
  }
}

/* Free. */
for (iaa=0; iaa<11; ++iaa) { for (ivv=0; ivv<41; ++ivv) { free(MKTA[ivv][iaa]); } }
free(mktwave);
return;
}



/* ----------------------------------------------------------------------
  Construct atmospheric absorption data.  -tab 01mar2018 

https://www.gemini.edu/sciops/telescopes-and-sites/observing-condition-constraints/ir-transmission-spectra
mktrans_zm_(mm*10)_(airmass*10).dat  at Mauna Kea
mktrans_zm_16_15.dat  would be 1.6mm water vapor column  and 1.5 airmass ..

12 original files:
"mktrans_zm_10_10" "mktrans_zm_10_15" "mktrans_zm_10_20"
"mktrans_zm_16_10" "mktrans_zm_16_15" "mktrans_zm_16_20"
"mktrans_zm_30_10" "mktrans_zm_30_15" "mktrans_zm_30_20"
"mktrans_zm_50_10" "mktrans_zm_50_15" "mktrans_zm_50_20"

*/
void nsx_construct_atmabs_data( double airmass, IMGtype IMG, SPXtype SPX[] )
{
/**/
int nn,order,ii,mktn,ecol,icol,nso,aa,vv,kk,wv,wvA,wvB;
int awvA[4],awvB[4];
/**/
double disp,xv,rr,wavemin,wavemax,sum,num,wave,tranA,tranB,tranxa,tranx;
double wtA,wtB,taux,tauA,tauB;
double coef[9],xoff,x8[3000],y8[3000],w8[3000];
/**/
FILE *infu;
FILE *infuA;
FILE *infuB;
FILE *outfu;
FILE *binfuW;
FILE *binfu3;
FILE *binfu4;
FILE *binfu5;
FILE *binfu6;
FILE *binfu7;
/**/
char wrd[200];
char wrdA[200];
char wrdB[200];
char lineA[2000];
char lineB[2000];
char line[2000];
char outfile[200];
/**/
const double ee = 2.7182818284590452;
/**/
double *mktx,*mkty,*mktys[9],*mkfw[9];
/**/
MKTTtype *MKTT;
/**/

/* Allocate. */
mktx = (double *)calloc((NUMMKT),sizeof(double));
mkty = (double *)calloc((NUMMKT),sizeof(double));
for (nso=3; nso<=7; ++nso) {
  mkfw[nso]  = (double *)calloc((NUMMKT),sizeof(double));
  mktys[nso] = (double *)calloc((NUMMKT),sizeof(double));
}
MKTT = (MKTTtype *)calloc((NUMMKT),sizeof(MKTTtype));

/* Set. */
aa = cnint(airmass*10.);


/* SKIP the .dat and .tbl creation steps? */
if (CONSTRUCT_ATMABS_DAT_TBL) {

/* arrays */
awvA[0]=10; awvB[0]=16;
awvA[1]=16; awvB[1]=30;
awvA[2]=30; awvB[2]=50;

/* Just scale the airmass=1.5 data file to the given airmass */
/* Create files for a range of water vapor.. */
/* Amm to Bmm .. */
for (kk=0; kk<3; ++kk) {
  wvA = awvA[kk];
  wvB = awvB[kk];
  printf("Range: wv= %d to %d\n",wvA,wvB);
  sprintf(wrdA,"/home/tb/nsx/cal/mktrans/mktrans_zm_%2.2d_%2.2d.dat",wvA,15);
  sprintf(wrdB,"/home/tb/nsx/cal/mktrans/mktrans_zm_%2.2d_%2.2d.dat",wvB,15);
  printf("A: %s  B: %s\n",wrdA,wrdB);
  for (wv=wvA; wv<=wvB; ++wv) {
    sprintf(wrd,"mktrans_zm_%2.2d_%2.2d.dat",wv,aa);
    printf("writing '%s'\n",wrd);
    outfu = fopen_write(wrd);
    infuA = fopen_read(wrdA);
    infuB = fopen_read(wrdB);
    while (fgetline(lineA,infuA)) {
      fgetline(lineB,infuB);
      wave  = GLV(lineA,1);
      if (wave < 2.51) {
        tranA = GLV(lineA,2);
        if (tranA < 1.e-12) { tranA = 1.e-12; }
        tauA  = -1.* log( tranA );
        tranB = GLV(lineB,2);
        if (tranB < 1.e-12) { tranB = 1.e-12; }
        tauB  = -1.* log( tranB );
        wtA   = wvB - wv;
        wtB   = wv  - wvA;
        taux  = ( (tauA * wtA) + (tauB * wtB) ) / (wtA + wtB);
        tranx = pow(ee,(-1*taux));
        tranxa= pow( tranx,(airmass/1.5) );
        fprintf(outfu,"%0.6f %0.6f\n",wave,tranxa);
      }
    }
    fclose(infuA);
    fclose(infuB);
    fclose(outfu);
  }
}


/* Read in all 'mktrans_zm_' files -- Mauna Kea Transmission at air=## and water vaper column ##mm */ 
/* Create NIRES echelle order smoothed transmissions ASCII table file. */
for (vv=10; vv<=50; vv=vv+1) {
sprintf(outfile,"mkts_%2.2d_%2.2d.tbl",vv,aa);

/* Read data. */
  sprintf(wrd,"mktrans_zm_%2.2d_%2.2d.dat",vv,aa);  printf(".........................read '%s'\n",wrd);
  infu = fopen_read(wrd);
  mktn=0;
  while (fgetline(line,infu)) {
    mktx[mktn] = GLV(line,1) * 10000.;   /* angstroms */
    mkty[mktn] = GLV(line,2);            /* Normalized transmission (1.0 = no absorption) */
    ++mktn;
    if (mktn > NUMMKT) { printf("***error: too many mktrans entries.\n"); exit(1); }
  }
  fclose(infu);

/* Convolve for each echelle order. */
  for (nso=3; nso<=7; ++nso) {
    if (nso == 7) { ecol=IMG.nc/2; } else { ecol=IMG.nc; }
/* Fit dispersion with 2nd order. */
    for (icol=0; icol<ecol; ++icol) {
      x8[icol] = SPX[nso].spwav[icol];   /* in angstroms */
      y8[icol] = SPX[nso].spdsp[icol];   /* in angstroms per pixel */
      w8[icol] = 1.0;
    }
    order = 2;
    if (GJ_polyfit(ecol,x8,y8,w8,order,0,&xoff,coef) != 1) { printf("***error:wave,disp fit failed.\n"); exit(1); }
/* Range. */
    rr = (SPX[nso].spwav[0] - SPX[nso].spwav[ecol-1]);
    wavemin = ( SPX[nso].spwav[ecol-1] ) - 100.;
    wavemax = ( SPX[nso].spwav[0]      ) + 100.;
    printf("nso=%d :  wavelength range: %f %f\n",nso,wavemin,wavemax);
/* Load FWHMs. */
    sum=0.; num=0.;
    for (ii=0; ii<mktn; ++ii) {
      if ((mktx[ii] < wavemin)||(mktx[ii] > wavemax)) {
        mkfw[nso][ii] = 0.;
      } else {
        xv = mktx[ii] - xoff;
        disp = cpolyval(order+1,coef,xv);            /* in angstroms per pixel */
        mkfw[nso][ii] = -1. * disp * PIXFWHM;
        sum = sum + mkfw[nso][ii];
        num = num + 1.;
      }
      mktys[nso][ii] = 0.;
    }
    if (num < 1.) exit(1);
    sum = sum / num;
    printf("convolve.. average fwhm=%f\n",sum);
    math_convolve_gaussian_fwhm( mktn, mktx, mkty, mkfw[nso], 4.0, mktys[nso] );
  }

/* Write data file. */
  outfu = fopen_write(outfile); printf("...............write %s\n",outfile);
  fprintf(outfu,"| wave    | transm  | smooth3 | fwhm3 | smooth4 | fwhm4 | smooth5 | fwhm5 | smooth6 | fwhm6 | smooth7 | fwhm7 |\n");
/*                123456789 123456789 123456789 1234567 123456789 1234567 */
  for (ii=0; ii<mktn; ++ii) {
    fprintf(outfu," %9.3f %9.6f %9.6f %7.4f %9.6f %7.4f %9.6f %7.4f %9.6f %7.4f %9.6f %7.4f \n",mktx[ii],mkty[ii],
         mktys[3][ii],mkfw[3][ii], mktys[4][ii],mkfw[4][ii], mktys[5][ii],mkfw[5][ii],
         mktys[6][ii],mkfw[6][ii], mktys[7][ii],mkfw[7][ii]);
  }
  fclose(outfu);

}

}


/* Write MKT binary files (using mkts*.tbl files). */

/* Read in all 'mkts_' files -- Mauna Kea Transmission at air=## and water vaper column ##mm */ 
printf("...............write binary files for airmass (aa=%d).\n",aa);
sprintf(wrd,"mkts_%2.2d_W.bin",aa); binfuW = fopen_write(wrd);
sprintf(wrd,"mkts_%2.2d_3.bin",aa); binfu3 = fopen_write(wrd);
sprintf(wrd,"mkts_%2.2d_4.bin",aa); binfu4 = fopen_write(wrd);
sprintf(wrd,"mkts_%2.2d_5.bin",aa); binfu5 = fopen_write(wrd);
sprintf(wrd,"mkts_%2.2d_6.bin",aa); binfu6 = fopen_write(wrd);
sprintf(wrd,"mkts_%2.2d_7.bin",aa); binfu7 = fopen_write(wrd);
for (vv=10; vv<=50; vv=vv+1) {

/* Clear. */
  for (ii=0; ii<NUMMKT; ++ii) {
    MKTT[ii].wave   =0.;
    MKTT[ii].tran[0]=0.;
    MKTT[ii].tran[1]=0.;
    MKTT[ii].tran[2]=0.;
    MKTT[ii].tran[3]=0.;
    MKTT[ii].tran[4]=0.;
  }

/* Read smoothed tables. */
  sprintf(wrd,"mkts_%2.2d_%2.2d.tbl",vv,aa); printf("Converting to binary.. read %s\n",wrd);
  infu = fopen_read(wrd);
  fgetline(line,infu);
  nn=0;
  while (fgetline(line,infu)) {
    MKTT[nn].wave   = GLV(line,1);
    MKTT[nn].tran[0]= GLV(line,3);
    MKTT[nn].tran[1]= GLV(line,5);
    MKTT[nn].tran[2]= GLV(line,7);
    MKTT[nn].tran[3]= GLV(line,9);
    MKTT[nn].tran[4]= GLV(line,11);
    ++nn;
    if (nn > NUMMKT) { printf("***error: too many entries in %s\n",wrd); exit(1); }
  }
  if (nn != NUMMKT) { printf("***error: nn=%d  NUMMKT=%d\n",nn,NUMMKT); exit(1); }
  printf("nn=%d\n",nn);

/* Write binary files. */
  for (ii=0; ii<nn; ++ii) { fwrite((char *)&MKTT[ii].wave,    1, sizeof(double), binfuW ); }
  for (ii=0; ii<nn; ++ii) { fwrite((char *)&MKTT[ii].tran[0], 1, sizeof(double), binfu3 ); }
  for (ii=0; ii<nn; ++ii) { fwrite((char *)&MKTT[ii].tran[1], 1, sizeof(double), binfu4 ); }
  for (ii=0; ii<nn; ++ii) { fwrite((char *)&MKTT[ii].tran[2], 1, sizeof(double), binfu5 ); }
  for (ii=0; ii<nn; ++ii) { fwrite((char *)&MKTT[ii].tran[3], 1, sizeof(double), binfu6 ); }
  for (ii=0; ii<nn; ++ii) { fwrite((char *)&MKTT[ii].tran[4], 1, sizeof(double), binfu7 ); }

}
fclose(binfuW);
fclose(binfu3);
fclose(binfu4);
fclose(binfu5);
fclose(binfu6);
fclose(binfu7);

return;
}




/* ----------------------------------------------------------------------
  Do flux calibration on vega-like star.
*/
void nsx_vega_flux_calibration( IMGtype IMG, SPXtype SPX[], char nsxdir[] )
{
/**/
const int VCmax = 2000;
double ratio[3000],CalFlux[3000],VegaScale,wave,VCx[VCmax],VCy[VCmax];
double ff[3000],xx[3000],yy[3000],ww[3000],scr[3000],wgt[3000],xv,coef[9],xoff;
double rms,median,arr[300];
/**/
int iicol,icol1,icol2,icol,ecol,nso,ii,VCn,ffnum[3000];
int kk,order,nn,ii1,ii2,narr,iii;
/**/
char wrd[200],root[80];
/**/
FILE *outfu;
/**/

/* Find this star. */
VegaScale = 0.;
if ((ABS((IMG.ra - 110.8090)) < 0.01)&&(ABS((IMG.dec - 15.8320)) < 0.01)) {
  printf("Star01: BD+16 1464..\n");
  strcpy(root,"Star01");
  VegaScale = 4384.;    /* (BD+16 1464 V=9, A1V C ~) */
}
if ((ABS((IMG.ra - 57.6334)) < 0.01)&&(ABS((IMG.dec - 29.7449)) < 0.01)) {
  printf("Star02: HD 24000..\n");
  strcpy(root,"Star02");
  VegaScale = 1542.;    /* (HD 24000 V=9, A0V C ) */
}
if (VegaScale < 0.0001) { printf("===warning: flux star not found.\n"); exit(0); }
nsx_load_vega( nsxdir, &VCn, VCx, VCy, VCmax ); printf("Load Vega: VCn=%d\n",VCn);

/* Each echelle order. */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IMG.nc/2; } else { ecol=IMG.nc; }

/* Load some values. */
  for (icol=0; icol<ecol; ++icol) {
    wave = SPX[nso].spwav[icol];
    kk = cneari_bs( wave, VCn, VCx );
    CalFlux[icol] = VCy[kk] / VegaScale;
    if (SPX[nso].spoac[icol] > 0.) { ratio[icol] = CalFlux[icol] / SPX[nso].spoac[icol]; } else { ratio[icol] = 0.; }
    ratio[icol] = ratio[icol] * 1.e+17;
    ff[icol]=0.; wgt[icol]=0.; ffnum[icol]=0;
  }

/* Fit curves. */
  for (icol=250; icol<ecol; icol=icol+125) {
    icol1 = icol - 250;
    icol2 = icol + 250; if (icol2 >= ecol-120) icol2=ecol-1;
    printf("nso=%d  icol1=%5d  icol2=%5d  range=%d : %9.2f %9.2f \n",
            nso,icol1,icol2,icol2-icol1,SPX[nso].spwav[icol1],SPX[nso].spwav[icol2]);
    if (icol2-icol1 < 80) { printf("***error: range too small.\n"); exit(1); }

/* Load. */
    nn=0;
    for (ii=icol1; ii<=icol2; ++ii) {
      xx[nn] = (double)ii;
      yy[nn] = ratio[ii];
      ww[nn] = 1.0;
      if (yy[nn] < 0.01) { ww[nn]=0.; }
      ++nn;
    }

/* Spike filter. */
    for (ii=0; ii<nn; ++ii) {
      ii1=ii-10; if (ii < 0  ) ii1=0;
      ii2=ii+10; if (ii >= nn) ii2=nn-1;
      narr=0; for (iii=ii1; iii<=ii2; ++iii) { arr[narr]=yy[iii]; ++narr; }
      if (narr < 4) { printf("***error narr=%d\n",narr); exit(1); }
      median = cfind_median8(narr,arr);
      rms=0.; for (iii=ii1; iii<=ii2; ++iii) { rms = rms + ((yy[iii] - median) * (yy[iii] - median)); }
      rms = sqrt(( rms / (double)narr ));
      if (ABS((yy[ii]-median)) > 1.5*rms) { ww[ii]=0.; }
    }

/* Fit. */
    order=3;
    if (GJ_polyfit(nn,xx,yy,ww,order,0,&xoff,coef) != 1) { printf("***error:vegacal fit failed.\n"); exit(1); }
    for (ii=0; ii<nn; ++ii) {
      iicol = cnint(xx[ii]);
      xv = xx[ii] - xoff;
      ff[iicol] = ff[iicol] + cpolyval(order+1,coef,xv);   
      wgt[iicol]= wgt[iicol]+ ww[ii];
      ++ffnum[iicol];
    }

  }

/* Normalize. */
  for (icol=0; icol<ecol; ++icol) {
    if (ffnum[icol] < 1) { printf("***error: ffnum too small.\n"); exit(1); }
    ff[icol] = ff[icol]  / (double)ffnum[icol];
    wgt[icol]= wgt[icol] / (double)ffnum[icol];
  }

/* Smooth fit. */
  nsx_SmoothArray8( 30, ecol, ff, scr );

/* Write. */
  sprintf(wrd,"vc%s_%d.tbl",root,nso); printf("Write '%s'.\n",wrd);
  outfu = fopen_write(wrd);
  fprintf(outfu,"| col | wave     | spobj      | spoac      | CalFlux    | ratio      | spatm  | ff         |wgt|\n");
/*                12345 1234567890 123456789012 123456789012 123456789012 123456789012 12345678 123456789012 123 */
  for (icol=0; icol<ecol; ++icol) {
    fprintf(outfu," %5d %10.3f %12.5e %12.5e %12.5e %12.5e %8.5f %12.5e %3.1f \n",
         icol, SPX[nso].spwav[icol], SPX[nso].spobj[icol], SPX[nso].spoac[icol],
         CalFlux[icol], ratio[icol], SPX[nso].spatm[icol], ff[icol], wgt[icol] );
  }
  fclose(outfu);
}

return;
}





/* ----------------------------------------------------------------------
  Extract object and background spectrum.
  Also apply wavelength scale.  Note use of EPERDN value.
*/
void nsx_extract_spectrum( IMGtype IMG, IMGtype IMGSKY, AVPtype AVP[],
                           SPXtype SPX[], char nsxdir[], char nsxout[] )
{
/**/
int ecol,nso,icol;
int pixno,kk,ii,jj,jjoff;
/**/
double airmass,as,sum,npx,rblo,rbhi,rb1,rb2;
double edge,sum_corimg,sum_varimg;
/**/
float *image;
/**/
MKTtype *MKT;
double *mktwave;
/**/

printf("Extracting spectrum.\n");

/* Load sky spectrum(.spsky[]), load (shifted) wavelength scale(.spwav[]), and dispersion(.spdsp[]). */
nsx_compute_skyline_shift( IMGSKY, SPX, nsxdir, nsxout );

/* Allocate, create background subtracted image. */
image = (float *)calloc((IMG.nc*IMG.nr),sizeof(float));
mktwave = (double *)calloc((NUMMKT),sizeof(double));
MKT = (MKTtype *)calloc((NUMMKT),sizeof(MKTtype));


/* Create background subtracted image. */
for (ii=0; ii<IMG.nc; ++ii) {
for (jj=0; jj<IMG.nr; ++jj) {
  pixno = ii + (jj * IMG.nc);
  image[pixno] = IMG.corimg[pixno] - IMG.bckimg[pixno];
}}


/* Construction of atmospheric absorption data */
if (CONSTRUCT_ATMABS) {
  printf("construct atmospheric absorption data..\n");
  for (airmass=1.0; airmass<2.01; airmass=airmass+0.1) {
    nsx_construct_atmabs_data( airmass, IMG, SPX );
  }
  printf("early exit\n");
  exit(0);
}


/* Extract object spectrum from corrected, background subtracted object image for each order. */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IMG.nc/2; } else { ecol=IMG.nc; }
  jjoff = ((7-nso) * 200);
  rblo= (double)jjoff;
  rbhi= (double)(jjoff + SPX[nso].numpro);
  rb1 = (double)jjoff + (SPX[0].asp1[0] / ARCSEC_PER_PIXEL);
  if (rb1 < rblo) rb1 = rblo;
  if (rb1 > rbhi) rb1 = rbhi;
  rb2 = (double)jjoff + (SPX[0].asp2[0] / ARCSEC_PER_PIXEL);
  if (rb2 < rblo) rb2 = rblo;
  if (rb2 > rbhi) rb2 = rbhi;
  as  = (SPX[0].asp1[0] + SPX[0].asp2[0]) / 2.;
  for (icol=0; icol<ecol; ++icol) {
    SPX[nso].spobj[icol] = nsx_fractional_pixel_rb( IMG.nc, image, rb1, rb2, icol );
    sum_corimg           = nsx_fractional_pixel_rb( IMG.nc, IMG.corimg, rb1, rb2, icol );
    sum_varimg           = nsx_fractional_pixel_rb( IMG.nc, IMG.varimg, rb1, rb2, icol );
    if (sum_varimg > 0.) { SPX[nso].sperr[icol] = sqrt(sum_varimg/EPERDN); } else { SPX[nso].sperr[icol] = 0.; }
    edge = nsx_find_real_image_row( 1, icol, nso );
    SPX[nso].sprow[icol] = edge + nsx_AVPinv(AVP,nso,icol,as,IMG);      /* Row position in original image. */
/* Scale to per second. */
    SPX[nso].spobj[icol] = SPX[nso].spobj[icol] / IMG.exptime;
    SPX[nso].sperr[icol] = SPX[nso].sperr[icol] / IMG.exptime;
  }
  SPX[nso].numsp = ecol;
}
/* -------------------------  -tab 09feb2018
 I noticed that the AVP polynomials have slight 'notches' in them when you plot offset
 vs. column.. but the notches are small (~<0.1pixels) so probably not significant..
 you can see the affect in the following outputs: 
  sprintf(wrd,"jaf%d.dat",nso);
  outfu = fopen_write(wrd);
  as = (SPX[0].asp1[0] + SPX[0].asp2[0]) / 2.;
  rowoff = as / ARCSEC_PER_PIXEL;  /x this is approx. since AVP refers to the uncorrected image x/
  for (icol=0; icol<ecol; ++icol) {
    rj1 = nsx_AVPinv(AVP,nso,icol,as,IMG);
    rj2 = nsx_AVP(AVP,nso,icol,rowoff,IMG);
    fprintf(outfu,"%d %f %f\n",icol,rj1,rj2);
  }
  fclose(outfu);
  -------------------------- */



/* Decide on best atmospheric transmission data set. */
if (ATMABS_CORRECTION) {
  nsx_find_best_mktfile( IMG, SPX, MKT, mktwave, nsxdir );
}

/* Vega comparison */
if (VEGA_FLUX_CALIBRATION) {
  nsx_vega_flux_calibration( IMG, SPX, nsxdir );
  if (nso != 43234) exit(0);
}


/* Extract background from corrected object image for each order (in background window(s)). */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IMG.nc/2; } else { ecol=IMG.nc; }
  jjoff = ((7-nso) * 200);
  rblo = (double)jjoff;
  rbhi = (double)(jjoff + SPX[nso].numpro);
  for (icol=0; icol<ecol; ++icol) {
    sum=0.; npx=0.;
    for (kk=0; kk<SPX[0].nbk; ++kk) {
      rb1 = (double)jjoff + (SPX[0].abk1[kk] / ARCSEC_PER_PIXEL);
      if (rb1 < rblo) rb1 = rblo;
      if (rb1 > rbhi) rb1 = rbhi;
      rb2 = (double)jjoff + (SPX[0].abk2[kk] / ARCSEC_PER_PIXEL);
      if (rb2 < rblo) rb2 = rblo;
      if (rb2 > rbhi) rb2 = rbhi;
      sum = sum + nsx_fractional_pixel_rb( IMG.nc, IMG.corimg, rb1, rb2, icol );
      npx = npx + ABS((rb2 - rb1));
    }
    if (npx > 0.) { SPX[nso].spbck[icol] = sum/npx; } else { SPX[nso].spbck[icol] = 0.; }
    SPX[nso].spbck[icol] = SPX[nso].spbck[icol] / IMG.exptime;
  }
}


/* Free. */
free(image);
free(MKT);
free(mktwave);
return;
}




/* ----------------------------------------------------------------------
  Create background image from corrected image. 
*/
void nsx_create_background_image( IMGtype IMG, AVPtype AVP[], SPXtype SPX[] )
{
/**/
int ecol,nso,hiii,jjoff,icol;
int pixno,ii,jj,reject,maxiter,iter,nn,asnum,kk,order;
/**/
double hisigs,devrms,devsigs,sigs,devmed,hidev,yrms,xv;
double ff,rblim,rb1,rb2,imgsum,asinc,asmax,as;
double xx[300],yy[300],ww[300],coef[9],xoff;
/**/

/* For 2017 calibration, I assumed 0.2 arcsec per pixel and use nso=3 width. */
asinc = ARCSEC_PER_PIXEL;
asnum = SPX[3].numpro - 2;
asmax = asinc * (double)asnum;
printf("Create background image using asinc=%f  asmax=%f  asnum=%d\n",asinc,asmax,asnum);

/* Clear. */
for (ii=0; ii<IMG.nc; ++ii) {
for (jj=0; jj<IMG.nr; ++jj) { pixno = ii + (jj * IMG.nc); IMG.bckimg[pixno] = 0.; }}

/* Sky fit on corrected image. */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IMG.nc/2; } else { ecol=IMG.nc; }
  jjoff = ((7-nso) * 200);
  for (icol=0; icol<ecol; ++icol) {

/* Load background pixels. */
    nn=0;
    for (kk=0; kk<SPX[0].nbk; ++kk) {
      rblim = (double)jjoff + (SPX[0].abk2[kk] / asinc);
      for (as=SPX[0].abk1[kk]; as<SPX[0].abk2[kk]; as=as+asinc) {
        rb1 = (double)jjoff + (as / asinc);
        rb2 = rb1 + 1.;
        if (rb2 > rblim) rb2=rblim;
        imgsum = nsx_fractional_pixel_rb( IMG.nc, IMG.corimg, rb1, rb2, icol );
        xx[nn] = (rb1 + rb2) / 2.;
        yy[nn] = imgsum;
        ww[nn] = rb2 - rb1;
        ++nn;
      }
    }

/* Fit pixels. */
    order = 2;
    reject= 1;
    iter  = 1;
    maxiter = nn/5;
    yrms=0.; devmed=0.; devrms=0.; hidev=0.; hiii=-1; sigs=0.; devsigs=0.; hisigs=0.;
    while ((reject)&&(iter < maxiter)) {
      if (GJ_polyfit(nn,xx,yy,ww,order,0,&xoff,coef) != 1) { printf("***error:cbi fit failed.\n"); exit(1); }
      nsx_hidevrms( nn, xx, yy, ww, order, xoff, coef, &yrms, &devmed, &devrms, &hidev, &hiii );
      if ((yrms > 0.)&&(devrms > 0.)&&(hiii >= 0)) { 
        sigs   = hidev / yrms; 
        devsigs= devmed / devrms; 
        hisigs = hidev  / devrms; 
      } else { sigs=0.; devsigs=0.; hisigs=0.; }
      if ((sigs > 5.)||(hisigs > 5.)) { ww[hiii]=0.; reject=1; } else { reject=0; }
      ++iter;
    }

/* Load background image. */
    for (jj=jjoff; jj<(jjoff+SPX[nso].numpro); ++jj) {
      xv = (double)jj - xoff;
      ff = cpolyval(order+1,coef,xv);
      pixno = icol + (jj * IMG.nc);
      IMG.bckimg[pixno] = ff;
    }

  }
}


return;
}


/* ----------------------------------------------------------------------
  Compute a sum of fractional pixel trapezoids.
  rba = bottom row,   rbb = top row
  cb1b= top row left column,    cb2b= top row right column, 
  cb1a= bottom row left column, cb2a= bottom row right column, 
 ------------   -------
 |  1b  2b  |   | 2 3 |
 |  1a  2a  |   | 0 1 |
 ------------   -------
*/
double nsx_fractional_trapezoid( int nc, float image[], double rba, double rbb, 
                                 double cb1a, double cb2a, double cb1b, double cb2b )
{
/**/
int cc,dd,ii[4],jj[4],kk,pixno,ok;
/**/
double height,slope,top,right,lower,upper,unkcol,left,bottom;
double area[4],img[4];
double imgsum = 0.;
double col[4],row[4];
/**/

/* Corners. */
row[0]=rba; col[0]=cb1a;
row[1]=rba; col[1]=cb2a;
row[2]=rbb; col[2]=cb1b;
row[3]=rbb; col[3]=cb2b;
for (kk=0; kk<=3; ++kk) { ii[kk]=cnint((col[kk])); jj[kk]=cnint((row[kk])); }

/* Check that we have 4 different pixels. */
ok=1;
for (cc=0; cc<=2; ++cc) {
for (dd=cc+1; dd<=3; ++dd) {
  if (ii[cc] == ii[dd]) {
    if (jj[cc] == jj[dd]) { ok=0; }
  }
}}
if (ok == 0) { printf("FATAL ERROR: corners not all different.\n"); exit(1); }

/* Check bottom-to-top and left-to-right */
if (ii[0] != ii[2]) ok=0;
if (ii[1] != ii[3]) ok=0;
if (jj[0] != jj[1]) ok=0;
if (jj[2] != jj[3]) ok=0;
if (ok == 0) { printf("FATAL ERROR: corners not lined up correctly.\n"); exit(1); }

/*
 ------------   -------
 |  1b  2b  |   | 2 3 |
 |  1a  2a  |   | 0 1 |
 ------------   -------
*/

/* 0: Bottom left corner pixel trapezoid. */
kk      = 0;
top     = (double)jj[kk] + 0.5;
right   = (double)ii[kk] + 0.5;
height  = top - rba;
lower   = right - cb1a;
slope   = (rbb - rba) / (cb1b - cb1a);      /* slope = (top - rba) / (unkcol - cb1a) */
unkcol  = ((top - rba) / slope) + cb1a;
upper   = right - unkcol;
area[kk]= height * (lower + upper) / 2.;
pixno   = ii[kk] + (jj[kk] * nc); 
img[kk] = image[pixno];

/* 1: Bottom right corner pixel trapezoid. */
kk      = 1;
top     = (double)jj[kk] + 0.5;
left    = (double)ii[kk] - 0.5;
height  = top - rba;
lower   = cb2a - left;
slope   = (rbb - rba) / (cb2b - cb2a);      /* slope = (top - rba) / (unkcol - cb2a) */
unkcol  = ((top - rba) / slope) + cb2a;
upper   = unkcol - left;
area[kk]= height * (lower + upper) / 2.;
pixno   = ii[kk] + (jj[kk] * nc); 
img[kk] = image[pixno];

/* 2: Upper left corner pixel trapezoid. */
kk      = 2;
bottom  = (double)jj[kk] - 0.5;
right   = (double)ii[kk] + 0.5;
height  = rbb - bottom;
upper   = right - cb1b;
slope   = (rbb - rba) / (cb1b - cb1a);      /* slope = (rbb - bottom) / (cb1b - unkcol) */
unkcol  = cb1b - ((rbb - bottom) / slope);
lower   = right - unkcol;
area[kk]= height * (lower + upper) / 2.;
pixno   = ii[kk] + (jj[kk] * nc); 
img[kk] = image[pixno];


/* 3: Top right corner pixel trapezoid. */
kk      = 3;
bottom  = (double)jj[kk] - 0.5;
left    = (double)ii[kk] - 0.5;
height  = rbb - bottom;
upper   = cb2b - left;
slope   = (rbb - rba) / (cb2b - cb2a);      /* slope = (rbb - bottom) / (cb2b - unkcol) */
unkcol  = cb2b - ((rbb - bottom) / slope);
lower   = unkcol - left;
area[kk]= height * (lower + upper) / 2.;
pixno   = ii[kk] + (jj[kk] * nc); 
img[kk] = image[pixno];

/* Sum. */
imgsum=0.;
for (kk=0; kk<=3; ++kk) { imgsum = imgsum + (area[kk] * img[kk]); }
return(imgsum);
}



/* ----------------------------------------------------------------------
  Correct the image (un-slant and un-curve).  -tab 19jan2018
  Also, clean the image first.
*/
void nsx_correct_image( IMGtype IMG[], SOPtype SOP1[], SOPtype SOP2[], AVPtype AVP[], SPXtype SPX[], 
                        int NoClean, int NoHotClean, char nsxdir[], int NoFlat )
{
/**/
int ecol,nso,jjoff,icol;
int asnum,ii,jj,pixno;
/**/
double edge,asinc,asmax,as;
double rr0,rr1,rowoff,rbave,rba,rbb,cb1,cb2,imgsum,imgsumNFD;
/**/
int nc = IMG[0].nc;   /* nominal 2048 */
int nr = IMG[0].nr;   /* nominal 1024 */
/**/
float *clnimgNFD;   /* clean image with No Flat Division */
float *fltimg;
/**/
char wrd[200];
/**/

/* Allocate. */
clnimgNFD = (float *)calloc(((nc*nr)+1000),sizeof(float));
fltimg    = (float *)calloc(((nc*nr)+1000),sizeof(float));

/* Clean the original image (remove CRs and hot spots). */
printf("Cleaning CRs and hot spots from image (%s).\n",IMG[0].root);
for (ii=0; ii<(nc*nr); ++ii) { IMG[0].clnimg[ii] = IMG[0].image[ii]; }
rr0 = timegetsec8();
nsx_clean_image( IMG, SPX, NoClean, NoHotClean, nsxdir );
rr1 = timegetsec8();
printf("... %f seconds\n",rr1-rr0);
for (ii=0; ii<(nc*nr); ++ii) { clnimgNFD[ii] = IMG[0].clnimg[ii]; }

/* Flat division. */
if (NoFlat == 0) {
  sprintf(wrd,"%scal/flatnorm-180304.fits",nsxdir);
  nsx_read_general_image( wrd, fltimg, nc, nr );
  for (ii=0; ii<(nc*nr); ++ii) { IMG[0].clnimg[ii] = IMG[0].clnimg[ii] / fltimg[ii]; }
  fprintf(logfu,"Completed flat field correction using '%s'.\n",wrd);
  printf(       "Completed flat field correction using '%s'.\n",wrd);
} else {
  fprintf(logfu,"No flat field correction has been done.\n");
  printf(       "No flat field correction has been done.\n");
}

/* Zero */
for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) {
  pixno = ii + (jj * nc);
  IMG[0].corimg[pixno] = 0.;
}}

/* For 2017 calibration, I used 0.2 arcsec per pixel and use nso=3 width. */
asinc = ARCSEC_PER_PIXEL;
asnum = SPX[3].numpro - 1;
asmax = asinc * (double)asnum;
printf("Correct image using asinc=%f  asmax=%f  asnum=%d\n",asinc,asmax,asnum);

/* Each order. */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  jjoff = ((7-nso) * 200);

/* Each column (where a 'column' starts at lower edge and then slants up). */
  for (icol=0; icol<ecol; ++icol) {
    edge = nsx_find_real_image_row( 1, icol, nso );
    for (as=0.; as<asmax; as=as+asinc) {

/* Select row (offset from edge) boundaries on arcsec scale. */
      rba = edge + nsx_AVPinv( AVP, nso, icol, as, IMG[0] );
      rbb = edge + nsx_AVPinv( AVP, nso, icol, as+asinc, IMG[0] );
      rbave = (rba + rbb) / 2.;
      rowoff = rbave - edge;

/* Column range using actual 'rowoff' value. */
      nsx_slant_boundaries_SOP2( nso, icol, rowoff, SOP1, SOP2, ecol, &cb1, &cb2 );

/* Fractional pixel. */
      nsx_fractional_pixel_2D( nc, IMG[0].clnimg, cb1, cb2, rba, rbb, &imgsum    );
      nsx_fractional_pixel_2D( nc, clnimgNFD    , cb1, cb2, rba, rbb, &imgsumNFD );
      ii = icol;
      jj = jjoff + cnint(( as / asinc ));
      pixno = ii + (jj * nc);
      IMG[0].corimg[pixno]    = imgsum;
      IMG[0].corimgNFD[pixno] = imgsumNFD;
    }
  }
}

/* Trim out first 12 columns and last 4 columns of corimg[]. */
for (jj=0; jj<nr; ++jj) { 
  for (ii=0;    ii<12; ++ii) { pixno=ii+(jj*nc); IMG[0].corimg[pixno]=0.; IMG[0].corimgNFD[pixno]=0.; }
  for (ii=nc-4; ii<nc; ++ii) { pixno=ii+(jj*nc); IMG[0].corimg[pixno]=0.; IMG[0].corimgNFD[pixno]=0.; }
}



nsx_write_general_image( "clnimg.fits", IMG[0].clnimg,  nc, nr );
nsx_write_general_image( "clnimgNFD.fits", clnimgNFD,  nc, nr );

nsx_write_general_image( "corimg.fits", IMG[0].corimg,  nc, nr );
nsx_write_general_image( "corimgNFD.fits", IMG[0].corimgNFD,  nc, nr );



free(clnimgNFD);
free(fltimg);
return;
}




/* ----------------------------------------------------------------------
  Mash profile on a arcsecond scale. 
  (no slant correction yet)
*/
void nsx_arcsec_profile_mash( IMGtype IMG, SPXtype SPX[], AVPtype AVP[] )
{
/**/
double peak,asmax,as,asinc,edge,sum,imgsum,rb1,rb2;
double apmx[300],apmave[300],apmmed[300];
double arr[MAXSP],guess,cent,radius;
/**/
int narr,nso,ecol,icol,ii,nn,niter;
/**/

/* Set and clear. */
/* For 2017 calibration, I used 0.2 arcsec per pixel and use nso=3 width. */
asinc = ARCSEC_PER_PIXEL;
asmax = cnint(( asinc * (double)SPX[3].numpro ));
nn=0;
for (as=0.; as<asmax; as=as+asinc) {
  SPX[0].pro_apx[nn] = as + (asinc/2.);
  SPX[0].pro_apymed[nn] = 0.;
  SPX[0].pro_apyave[nn] = 0.;
  ++nn;
}
SPX[0].pro_apn = nn;

/* Each order. ( Could change to 4..6 rather than 3..7 ? ). */
for (nso=3; nso<=7; ++nso) {
if (nso == 7) { ecol=IMG.nc/2; } else { ecol=IMG.nc; }
  nn=0;
  for (as=0.; as<asmax; as=as+asinc) {
    sum=0.; narr=0;
    for (icol=0; icol<ecol; ++icol) {
      edge = nsx_find_real_image_row( 1, icol, nso );
      rb1 = edge + nsx_AVPinv( AVP, nso, icol, as, IMG );
      rb2 = edge + nsx_AVPinv( AVP, nso, icol, as+asinc, IMG );
      imgsum = nsx_fractional_pixel_rb( IMG.nc, IMG.image, rb1, rb2, icol );
      sum = sum + imgsum;
      arr[narr] = imgsum;
      ++narr;
    }
    apmx[nn]  = as + (asinc/2.);
    apmave[nn]= sum / (double)narr;
    apmmed[nn]= cfind_median8(narr,arr);
    ++nn;
  }

/* Peak? */
  peak=0.0001; guess=0.;
  for (ii=0; ii<nn; ++ii) { 
    if (apmmed[ii] > peak) { peak=apmmed[ii]; guess=apmx[ii]; }
  }

/* Centroid */
  radius=4.; niter=3;
  cent = nsx_centroid2(nn,apmx,apmmed,guess,radius,niter,0);
  printf("Profiles: nso=%d  centroid=%9.3f  %9.3f(px) \n",nso,cent,cent/ARCSEC_PER_PIXEL);

/* Save. */
  for (ii=0; ii<nn; ++ii) {
    SPX[nso].pro_apx[ii]   = apmx[ii];
    SPX[nso].pro_apymed[ii]= apmmed[ii];
    SPX[nso].pro_apyave[ii]= apmave[ii];
    SPX[0].pro_apymed[ii]  = SPX[0].pro_apymed[ii] + apmmed[ii];
    SPX[0].pro_apyave[ii]  = SPX[0].pro_apyave[ii] + apmave[ii];
  }
  SPX[nso].pro_apn = nn;

}


return;
}












/* ----------------------------------------------------------------------
  Centroid to dip in the ppsum[] array to find more accurate offset value.
*/
double nsx_wpcent( int pp, int npp, double ppsum[] )
{
/**/
double wgt,cent,hi1,hi2,base,sum,wsum;
int ii,ii1,ii2;
/**/

/* Limits. */
ii1=pp-3; 
if (ii1 < 0    ) ii1=0;
if (ii1 > npp-1) ii1=npp-1;
ii2=pp+3;
if (ii2 < 0    ) ii2=0;
if (ii2 > npp-1) ii2=npp-1;

/* Find base level (lower of two sides. */
hi1=0.; for (ii=ii1; ii<=pp; ++ii) { if (ppsum[ii] > hi1) hi1=ppsum[ii]; }
hi2=0.; for (ii=pp; ii<=ii2; ++ii) { if (ppsum[ii] > hi2) hi2=ppsum[ii]; }
if (hi1 < hi2) { base=hi1; } else { base=hi2; }

/* Limits. */
ii1=pp-1; 
if (ii1 < 0    ) ii1=0;
if (ii1 > npp-1) ii1=npp-1;
ii2=pp+1;
if (ii2 < 0    ) ii2=0;
if (ii2 > npp-1) ii2=npp-1;

/* Centroid. */
sum=0.; wsum=0.;
for (ii=ii1; ii<=ii2; ++ii) {
  wgt = base - ppsum[ii];
  if (wgt > 0.) {
    sum = sum + ( wgt * (double)ii );
    wsum= wsum+ wgt;
  }
}
if (wsum > 0.) { cent = sum / wsum; } else { cent=(double)pp; }

return(cent);
}


/* ----------------------------------------------------------------------
  Wide mash of nso=3 ..    -tab 27apr2018
*/
void nsx_wide_profile_mash( IMGtype IMG )
{
/**/
double ppsum[300],base1,base2,base,sum,num,pro[300],pro_mdn[300];
double lolosum,losum[9],cutoff,lobase;
double SlitCenterI,SlitCenterD,wpcentroid[9];
/**/
float arr[(IMG.nc+10)];
/**/
int mw,ppp,pp2,eao,lopp[9],ii,jj,pp,pixno,ecol,npp;
int nso = 3;
/**/
int SearchAdd = 80;
int SearchAddHalf = SearchAdd / 2;
int ExtraAdd  = 4;
/**/
char wrd[200];
/**/

/* Set.  */
ecol = IMG.nc;
mw = nsx_minwidth(ecol,nso);
npp= mw + SearchAdd;

/* Build wide profile. */
for (pp=0; pp<npp; ++pp) {
  pro[pp] = 0.;
  pro_mdn[pp] = 0.;
  sum=0.; num=0.;
  for (ii=20; ii<ecol-20; ++ii) {
    jj = (pp - SearchAddHalf) + nsx_find_image_row( 1, ii, nso );
    if ((jj >= 0)&&(jj < IMG.nr)) {
      pixno = ii+(jj*IMG.nc);
      arr[cnint(num)] = IMG.image[pixno];
      sum = sum + IMG.image[pixno];
      num = num + 1.;
    }
  }
  if (num > 0.) { pro[pp] = sum / num; } else { pro[pp]=0.; }
  if (num > 2.) { pro_mdn[pp] = cfind_median(cnint(num),arr); } else { pro[pp]=0.; }
}


/* Look at bracket sum with a range of windows. */
lobase=0.; lolosum=1.e+10;
for (eao=8; eao>=0; --eao) {
  losum[eao]=1.e+10; lopp[eao]=-1000;
  for (pp=0; pp<npp-mw; ++pp) {
    pp2 = pp + mw + ExtraAdd + eao;
    if (pp2 < npp) {
      ppsum[pp] = pro_mdn[pp] + pro_mdn[pp2];
      if (ppsum[pp] < losum[eao]) { lopp[eao]=pp; losum[eao]=ppsum[pp]; }
    }
  }
/* Base level? */
  ppp=lopp[eao]-8; 
  if (ppp <  0 ) ppp=0;
  if (ppp >=npp) ppp=npp-1;
  base1=ppsum[ppp];
  ppp=lopp[eao]+8; 
  if (ppp <  0 ) ppp=0;
  if (ppp >=npp) ppp=npp-1;
  base2=ppsum[ppp];
  if (base1 < base2) { base=base1; } else { base=base2; }
/*
  printf("eao=%3d  losum[eao]=%7.2f  lopp[eao]=%3d   base=%f\n",eao,losum[eao],lopp[eao],base);
*/
  if (losum[eao] < lolosum) { lolosum=losum[eao]; lobase=base; }
  wpcentroid[eao] = nsx_wpcent( lopp[eao], npp, ppsum );
}
cutoff = (lolosum + lobase ) / 2.; 
if (cutoff < lolosum) { cutoff = lolosum*1.01; }
if (lolosum < 1.) lolosum=1.; 


/* Find smallest window where low point is still less than cutoff. */
eao=8;
while ((losum[eao] < cutoff)&&(eao >=-4)) { --eao; }
if (eao < 8) eao=eao+1;


/* Set global parameter for slit offset. */
if (lopp[eao] != -1000) {
  SlitCenterI     = ( (double)(lopp[eao]) + (double)(lopp[eao]+mw+ExtraAdd+eao) ) / 2.;
  SlitCenterD = ( wpcentroid[eao] + (double)(lopp[eao]+mw+ExtraAdd+eao) ) / 2.;
} else {
  SlitCenterI = 98.;
  SlitCenterD = 98.;
}
if (ABS((SlitCenterI-SlitCenterD)) > 0.5) {
  SlitOffset = SlitCenterI - 97.9;
} else {
  SlitOffset = SlitCenterD - 97.9;
}



/* comments? */
sprintf(wrd,"For '%s' the slit offset from nominal calibration is %9.5f rows.",IMG.file,SlitOffset);
printf("%s\n",wrd);
fprintf(logfu,"%s\n",wrd);


return;
}



/* ----------------------------------------------------------------------
 * OBSOLETE OLD ROUTINE *
  Calibrate curve of point source along slit as function of column.  -tab 11dec2017
  How to calibrate:
  - Clear (delete/touch) calibration file 'calcurve.dat'.
  - Set CALCURVE defined variable to '1'.
  - run 'nsx (image fits file of point source exposure)' repeatably on many point
      source exposures (10-100).
  - This loads point source curve info into calcurve.dat, this is the 'rowoff'
      value of the centroid of the point source vs. the 'arcsec' value.  The 'arcsec'
      value is defined as the col=1000 centroid of nso=3 where a pixel is 
      'ARCSEC_PER_PIXEL' (0.2) arcseconds.
      (The arcsec value is the same for all orders of a given exposure of course).
  - Thus 'rowoff' vs. 'arcsec' are accumulated for each column and each exposure.
  - Once calcurve.dat is complete, set CALCURVE2 to '1' and rerun to generate the 
      AVP polynomials which are fit for each column and nso.
*/
void nsx_calcurve( IMGtype IMG, SPXtype SPX[], AVPtype AVP[] )
{
/**/
int iter,hiii,order,ecol,nc,nro,ii,jj,pp,narr_as,narr,rowicol,icol,kk,nso,nn,pixno;
/**/
double cent,edge,guess,hi,arcsec,arcsec1000,rowoff,arcsecold;
double arr_as[100],arr[100],xx[MAXPRO],yy[MAXPRO];
double xv,fro,xoff,coef[9],high,wrms,wppd,xro[MAXSP],yro[MAXSP],wro[MAXSP];
/**/
char wrd[100];
FILE *outfu;
FILE *datfu;
/**/

/* First find 'arcsec' value at nso=3 col=1000. */
arcsec1000 = 0.;
nc  = IMG.nc;
narr_as= 0;
nso = 3;
nn  = SPX[nso].numpro;
for (icol=900; icol<=1100; icol=icol+50) {
  edge = nsx_find_real_image_row( 1, icol, nso );
  rowicol = cnint(edge);
  for (pp=0; pp<nn; ++pp) {
    jj = rowicol + pp;
    narr = 0;
    for (ii=icol-10; ii<=icol+10; ++ii) {
      pixno = ii+(jj*nc);
      arr[narr]= IMG.image[pixno];
      ++narr;
    }
    xx[pp] = (double)jj;
    yy[pp] = cfind_median8(narr,arr);
  }
/*
  sprintf(wrd,"junk%4.4d.dat",icol); printf("write %s\n",wrd);
  outfu = fopen_write(wrd);
  for (ii=0; ii<nn; ++ii) { fprintf(outfu,"%f %f\n",xx[ii],yy[ii]); }
  fclose(outfu);
*/
  guess=0.; hi=-9999.;
  for (ii=0; ii<nn; ++ii) { if (yy[ii] > hi) { hi=yy[ii]; guess=xx[ii]; } }
  cent = nsx_centroid2(nn,xx,yy,guess,5.,4,1);
  if (cent < -9000.) exit(1);
  rowoff = cent - edge;
  arcsecold = nsx_AVP( AVP, nso, rowicol, rowoff, IMG );
  arcsec = rowoff * ARCSEC_PER_PIXEL;
  if (icol == 1000) arcsec1000 = arcsec;
  printf("icol=%4d  rowicol=%4d  edge=%8.3f  cent=%8.3f   rowoff=%8.4f   arcsec=%8.4f  arcsecold=%8.4f\n",
       icol,rowicol,edge, cent,rowoff,arcsec,arcsecold);
  arr_as[narr_as] = arcsec;
  ++narr_as;
}
arcsec = cfind_median8(narr_as,arr_as);
printf("arcsec=%f  arcsec1000=%f  dif=%f \n",arcsec,arcsec1000,arcsec-arcsec1000);
if (ABS((arcsec-arcsec1000)) > 0.002) { 
  printf("===warning: arcsec not close enough (<0.002) to the 1000 value.. using %f\n",arcsec); cpauseit();
}


/* Now fit polynomials along echelle orders (with iterate rejections). */
misc_delete_file("temp.dat");
datfu = fopen_write("temp.dat");
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  nn = SPX[nso].numpro;
  nro= 0;
  for (icol=20; icol<ecol-12; icol=icol+20) {
    edge = nsx_find_real_image_row( 1, icol, nso );
    rowicol = cnint(edge);
    for (pp=0; pp<nn; ++pp) {
      jj = rowicol + pp;
      narr=0;
      for (ii=icol-10; ii<=icol+10; ++ii) {
        pixno = ii+(jj*nc);
        arr[narr]= IMG.image[pixno];
        ++narr;
      }
      xx[pp] = (double)jj;
      yy[pp] = cfind_median8(narr,arr);
    }
/*
    sprintf(wrd,"junk_%d_%4.4d.dat",nso,icol); printf("write %s\n",wrd);
    outfu = fopen_write(wrd);
    for (ii=0; ii<nn; ++ii) { fprintf(outfu,"%f %f\n",xx[ii],yy[ii]); }
    fclose(outfu);
*/
    guess=0.; hi=-9999.;
    for (ii=0; ii<nn; ++ii) { if (yy[ii] > hi) { hi=yy[ii]; guess=xx[ii]; } }
    cent = nsx_centroid2(nn,xx,yy,guess,5.,4,1);
    if (cent < -9000.) exit(1);
    xro[nro] = (double)icol;
    yro[nro] = cent - edge;
    wro[nro] = 1.0;
    ++nro;
  }
  order = 4;
  for (iter=0; iter<5; ++iter) {
    if (GJ_polyfit(nro,xro,yro,wro,order,0,&xoff,coef) != 1) { printf("***error:calcurve:fit failed.\n"); cpauseit(); }
    GJ_polyfit_residuals( nro, xro, yro, wro, order, xoff, coef, &high, &wrms, &wppd );
    printf("iter=%d nso=%d  icol=%4d : nro=%4d  high=%7.3f  wrms=%7.3f  wppd=%7.3f \n",
            iter,nso,icol,nro,high,wrms,wppd);
    hiii=-1; hi=0.;
    for (ii=0; ii<nro; ++ii) { if (wro[ii] > 0.) {
      xv = xro[ii] - xoff;
      fro= cpolyval(order+1,coef,xv);
      if (ABS((fro - yro[ii])) > hi) { hi=ABS((fro - yro[ii])); hiii=ii; }
    }}
    if (hiii > -1) { wro[hiii]=0.; }
  }
  kk = cnint(arcsec * 100.);
  sprintf(wrd,"junkf_%4.4d_%d.dat",kk,nso); printf("write %s\n",wrd);
  outfu = fopen_write(wrd);
  for (ii=0; ii<nro; ++ii) { 
    xv = xro[ii] - xoff;
    fro= cpolyval(order+1,coef,xv);
    fprintf(outfu,"%f %f %f \n",xro[ii],yro[ii],fro);
  }
  fclose(outfu);
  fprintf(datfu," %20.12e %2d %2d %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e \n",
     arcsec,nso,order,xoff,coef[0],coef[1],coef[2],coef[3],coef[4]);
}
fclose(datfu);
misc_delete_file("temp2.dat");
misc_system("touch calcurve.dat");
misc_system("cat calcurve.dat temp.dat > temp2.dat");
misc_system("mv temp2.dat calcurve.dat");

return;
}




/* ----------------------------------------------------------------------
 * OBSOLETE OLD ROUTINE *
  Once calcurve.dat is complete (CALCURVE), define CALCURVE2 to '1' and rerun to 
    generate the AVP polynomials which are fit for each column and nso and create
    the AVP.dat and AVPinv.dat files.   (see load AVP)
*/
void nsx_calcurve2( int nc )
{
/**/
int pp,orderi,nrej,nni,maxiter,order,nso,hijj,iter,ecol,icol,jj,nexp,ord4,kk;
/**/
const int mexp = 200;
double arcsecs[mexp],xoffs[mexp],coefs[mexp][9];
double ff,xv,xx[mexp],yy[mexp],ww[mexp],xoff,coef[9];
double dif,hidif,high,wrms,wppd;
double xxi[mexp],yyi[mexp],wwi[mexp],xoffi,coefi[9];
/**/
const double aspp = ARCSEC_PER_PIXEL;
/**/
char line[2000];
/**/
FILE *infu;
FILE *outfu3;
FILE *forfu;
FILE *invfu;
/**/

forfu = fopen_write("AVP.dat");
invfu = fopen_write("AVPinv.dat");

/* Read in data for each nso. */
outfu3 = fopen_write("calcurve2_res.tbl");
fprintf(outfu3,"|nso|icol|nrej|highpx|wrmspx|\n");
for (nso=3; nso<=7; ++nso) { if (nso > 0) {
  printf("calcurve2: nso=%d\n",nso);
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }

/* Read in all exposures with 'nso' data. */
  nexp=0;
  infu = fopen_read("calcurve.dat");
  while (fgetline(line,infu)) {
    kk = GLV(line,2);
    if (kk == nso) {
      ord4 = GLV(line,3);
      if (ord4 != 4) { printf("***error: order must be 4.\n"); exit(1); }
      arcsecs[nexp] = GLV(line,1);
      xoffs[nexp]   = GLV(line,4);
      coefs[nexp][0]= GLV(line,5);
      coefs[nexp][1]= GLV(line,6);
      coefs[nexp][2]= GLV(line,7);
      coefs[nexp][3]= GLV(line,8);
      coefs[nexp][4]= GLV(line,9);
      ++nexp;
      if (nexp > mexp-3) { printf("***error: too many exposures in calcurve.dat (change mexp).\n"); exit(1); }
    }
  }
  fclose(infu);

/* Fit to each column. */
  maxiter=4;
  if (nso > 5) maxiter=5;
  iter=1;  order=2;
  for (icol=0; icol<ecol; ++icol) { if (icol > -1900) {
    for (jj=0; jj<nexp; ++jj) {
      xv     = (double)icol - xoffs[jj];
      xx[jj] = cpolyval(ord4+1,coefs[jj],xv);
      yy[jj] = arcsecs[jj];
      ww[jj] = 1.0;
    }
    nrej=0;
    for (iter=0; iter<maxiter; ++iter) {
      if (GJ_polyfit(nexp,xx,yy,ww,order,0,&xoff,coef) != 1) { printf("***error:calcurve:fit failed.\n"); cpauseit(); }
      hidif=-9.; hijj=-1;
      for (jj=0; jj<nexp; ++jj) { if (ww[jj] > 0.) {
        xv = xx[jj] - xoff;
        ff = cpolyval(order+1,coef,xv);
        dif= ABS((yy[jj]-ff));
        if (dif > hidif) { hidif=dif; hijj=jj; }
      }}
      if ((hidif/aspp) > 0.25) { 
        ww[hijj]=0.; 
/*
        printf("Reject %7.3f xx=%7.3f   nso=%d  icol=%4d  iter=%d\n",hidif/aspp,xx[hijj],nso,icol,iter); 
*/
        ++nrej;
      }
    }
    GJ_polyfit_residuals( nexp, xx, yy, ww, order, xoff, coef, &high, &wrms, &wppd );
    if (ABS((high/aspp)) > 0.30) {
      printf("iter=%d nso=%d icol=%4d nexp=%2d high=%7.3f wrms=%7.3f wppd=%7.3f : high(px)=%7.3f  wrms(px)=%7.3f \n",
              iter,nso,icol,nexp,high,wrms,wppd,high/aspp,wrms/aspp);
    }
    fprintf(outfu3," %3d %4d %4d %6.3f %6.3f \n",nso,icol,nrej,high/aspp,wrms/aspp);
/*
    outfu = fopen_write("junk.dat");
    for (jj=0; jj<nexp; ++jj) {
      xv = xx[jj] - xoff;
      ff = cpolyval(order+1,coef,xv);
      fprintf(outfu," %9.3f %9.3f %9.3f %9.4f %9.4f %6.2f\n",xx[jj],yy[jj],ff,yy[jj]-ff,(yy[jj]-ff)/aspp,ww[jj]);
    }
    fclose(outfu);
*/

/* Write out AVP.dat and AVPinv.dat .. */
    fprintf(forfu," %3d %4d %20.12e %20.12e %20.12e %20.12e \n",nso,icol,xoff,coef[0],coef[1],coef[2]);

/* Fit inverse. */
    nni = 120;
    for (pp=0; pp<nni; ++pp) {
      xv = (double)pp - xoff;
      xxi[pp] = cpolyval(order+1,coef,xv);
      yyi[pp] = (double)pp;
      wwi[pp] = 1.0;
    }
    orderi = 2;
    if (GJ_polyfit(nni,xxi,yyi,wwi,orderi,0,&xoffi,coefi) != 1) { printf("***error:inv fit failed.\n"); exit(1); }
    GJ_polyfit_residuals( nni, xxi, yyi, wwi, orderi, xoffi, coefi, &high, &wrms, &wppd );
    fprintf(invfu," %3d %4d %20.12e %20.12e %20.12e %20.12e \n",
                    nso,icol,xoffi,coefi[0],coefi[1],coefi[2]);
    if (ABS((high)) > 0.02) {
      printf("inv: nso=%d icol=%4d : nni=%3d  high=%12.7f  wrms=%12.7f  wppd=%12.7f \n",nso,icol,nni,high,wrms,wppd);
      cpauseit();
    }

  }}

}}
fclose(outfu3);
fclose(forfu);
fclose(invfu);

return;
}





/* ----------------------------------------------------------------------
  Calibrate slant across slit.
  How to calibrate:
    Create raw sum file of emission line exposures (e.g. arcsum.fits).
    Set 'define CALSLANT 1' in code above.
    Run 'nsx arcsum.fits xsp=40,80' (for example).
    Start with nso=3, create blank slant(nso).dat file, run routine, program will
      exit early with 'sl_(nso)-000.tbl' file.
    Plot 'sl_' file and pick out emission lines, load line column numbers into
      the 'slant(nso).dat' file, one number per text file line.
    Run 'nsx arcsum.fits xsp=40,80' (again).
    Run 'pgim arcsum.fits draw=slant(nso).draw' to see centroids on top of emission lines.
    Run 'pgxy sl_(nso)-000.tbl draw=sl_(nso)-000.draw' to see guess,centroid,background fits of 
      lines on emission line plot.. repeat for sl_(nso)-010, sl_(nso)-020, etc..
    Look at 'efit_(nso)-##.tbl' to see fits of emission lines down slit, also look at
      nsx output with high,wrms,wppd results of poly fits. 
      ( try: ' pgxy xn=xx yn=yy zn=ff nh=4 nv=4 sch=2.2 efit_3-??.tbl ' )
      ( try: ' pg4 xn=xx yn=res nh=4 nv=4 min=-1 max=1 sch=2.2 efit_3-??.tbl ' )
    Look at 'rfit_(nso)-###.tbl' to see fits of offset values across columns for each row.
      ( try: ' pgxy xn=xx yn=yy zn=ff nh=4 nv=4 sch=2.2 rfit_3-??0.tbl ' )
      ( try: ' pg4 xn=xx yn=res nh=4 nv=4 min=-1 max=1 sch=2.2 rfit_3-??0.tbl ' )
    Finally look at 'offimg_(nso).fits to see offset image for given 'nso'.
    This 'offimg' file is the calibrator used for slanted extractions.
    (Then you must extract a wavelength calibrator using these 'offimg' data and re-calibrate
     the wavelength scale).
*/
void nsx_calslant( float image[], int nc, int nr )
{
/**/
int dd,nn,icol,irow,numlines,lower,nso,narr;
int bckn,order,ii,jj,pp,pixno,ecol,kk,mw;
/**/
float arr[200];
float *offimg;
float bckx[100],bcky[100];
/**/
double xx[4000],yy[4000],zz[4000],ww[4000];
double cent[100][100],row[100][100],guess[100];
double averow,ff,xv;
double xoff,coef[9];
double back,peak,high,wrms,wppd;
/**/
FILE *infu;
FILE *outfu;
FILE *drawfu;
/**/
char wrd[100];
char line[100];
char slantfile[100];
/**/

offimg = (float *)calloc(200*nc,sizeof(float));


for (nso=3; nso<=7; ++nso) {
sprintf(slantfile,"slant%d.dat",nso);
if (FileExist(slantfile)) {

printf("calslant on nso=%d ...\n",nso);

numlines=0;
if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
mw = nsx_minwidth(ecol,nso);
for (kk=0; kk<mw-10; kk=kk+10) {

  sprintf(wrd,"sl_%d-%3.3d.tbl",nso,kk);
  outfu = fopen_write(wrd);
  fprintf(outfu,"| col   | median  |\n");
  nn=0;
  for (ii=0; ii<ecol; ++ii) {
    lower = nsx_find_image_row( 1, ii, nso );
    narr=0; averow=0.;
    for (pp=0; pp<10; ++pp) {
      jj = lower + kk + pp;
      averow = averow + (double)jj;
      pixno = ii+(jj*nc);
      arr[narr] = image[pixno];
      ++narr;
    }
    xx[ii] = (double)ii;
    yy[ii] = averow / (double)narr;
    zz[ii] = cfind_median(narr,arr);
    ++nn;
    fprintf(outfu," %7.1f %9.2f \n",xx[ii],zz[ii]);
  }
  fclose(outfu);

  if (kk == 0) {
    numlines=0;
    infu = fopen_read(slantfile);
    while (fgetline(line,infu)) { guess[numlines]=GLV(line,1); ++numlines; }
    fclose(infu);
  }

/* Exit early if slant#.dat file blank. */
  if (numlines == 0) {
    printf("===warning: numlines=0 on nso=%d ... exit early ... look at '%s' to pick out lines and build up slant%d.dat file.\n",nso,wrd,nso);
    exit(0);
  }

  sprintf(wrd,"sl_%d-%3.3d.draw",nso,kk);
  drawfu = fopen_write(wrd);
  jj=kk/10;
  for (ii=0; ii<numlines; ++ii) {
    cent[ii][jj]= nsx_centroid3( nn,xx,zz,guess[ii],5.,3,&back,&peak, bckx,bcky,&bckn );
    row[ii][jj] = yy[ cnint(( cent[ii][jj] )) ];
    fprintf(drawfu,"sci 3\n");
    fprintf(drawfu," %9.3f %9.3f \n",guess[ii],back);
    fprintf(drawfu," %9.3f %9.3f \n",guess[ii],peak);
    fprintf(drawfu,"draw\n");
    guess[ii] = cent[ii][jj];
    fprintf(drawfu,"sci 2\n");
    fprintf(drawfu," %9.3f %9.3f \n",guess[ii],peak);
    fprintf(drawfu," %9.3f %9.3f \n",guess[ii],back);
    fprintf(drawfu,"draw\n");
    fprintf(drawfu,"sci 4\n");
    for (dd=0; dd<bckn; ++dd) { fprintf(drawfu," %9.3f %9.3f \n",bckx[dd],bcky[dd]); }
    fprintf(drawfu,"draw\n");
  }
  fclose(drawfu);

}


sprintf(wrd,"slant%d.draw",nso);
printf("Writing '%s'\n",wrd);
drawfu = fopen_write(wrd);
sprintf(wrd,"slants%d.out",nso);
outfu  = fopen_write(wrd);
fprintf(drawfu,"sci 3 ; sym 17 ; sch 2.0 \n");
for (kk=0; kk<mw-10; kk=kk+10) {
  jj=kk/10;
  for (ii=0; ii<numlines; ++ii) {
    fprintf(drawfu," %9.3f %9.3f \n",cent[ii][jj],row[ii][jj]);
    fprintf(outfu," %9.3f %9.3f \n",cent[ii][jj]-cent[ii][0],row[ii][jj]-row[ii][0]);
  }
}
fprintf(drawfu,"plot\n");
fclose(drawfu);
fclose(outfu);


/* Clear. */
for (icol=0; icol<nc; ++icol) {
for (irow=0; irow<200; ++irow) {
  pixno = icol + (irow * nc); 
  offimg[pixno] = -999.;
}}

/* Offset image. */
for (ii=0; ii<numlines; ++ii) {
  icol = cnint((cent[ii][0]));

/* Fit down each emission line. */
  nn=0;
  for (kk=0; kk<mw-10; kk=kk+10) {
    jj=kk/10;
    xx[nn] = row[ii][jj] - row[ii][0];
    yy[nn] = cent[ii][0] - cent[ii][jj];
    ww[nn] = 1.;
    ++nn;
  }
  order=2;
  if (GJ_polyfit(nn,xx,yy,ww,order,0,&xoff,coef) != 1) {
    printf("***error:calslant:fit failed.\n"); exit(1);
  }
  GJ_polyfit_residuals( nn, xx, yy, ww, order, xoff, coef, &high, &wrms, &wppd );
  printf("Line#%2d : nn=%d  high=%7.3f  wrms=%7.3f  wppd=%7.3f \n",ii,nn,high,wrms,wppd);

  sprintf(wrd,"efit_%d-%2.2d.tbl",nso,ii);
  outfu = fopen_write(wrd);
  fprintf(outfu,"| xx      | yy      | ff      | res     |\n");
  for (kk=0; kk<nn; ++kk) {
    xv = xx[kk] - xoff;
    ff = cpolyval(order+1,coef,xv);
    fprintf(outfu," %9.3f %9.3f %9.3f %9.3f \n",xx[kk],yy[kk],ff,yy[kk]-ff);
  }
  fclose(outfu);

/* Load values into offset image. */
  for (irow=0; irow<mw+10; ++irow) {
    pixno = icol + (irow * nc); 
    xv = (double)irow - xoff;
    offimg[pixno] = cpolyval(order+1,coef,xv);
  }

}


/* Fit across rows. */
for (irow=0; irow<mw+10; ++irow) {

  order=2; nn=0;
  for (icol=0; icol<ecol; ++icol) {
    pixno = icol + (irow * nc);
    if (offimg[pixno] > -900.) {
      xx[nn] = (double)icol;
      yy[nn] = (double)offimg[pixno];
      ww[nn] = 1.0;
      ++nn;
    }
  }

  if (GJ_polyfit(nn,xx,yy,ww,order,0,&xoff,coef) != 1) {
    printf("***error:calslant2:%3.3d:fit failed (nn=%d).\n",irow,nn); exit(1);
  }


  sprintf(wrd,"rfit_%d-%3.3d.tbl",nso,irow);
  outfu = fopen_write(wrd);
  fprintf(outfu,"| xx      | yy      | ff      | res     |\n");
  for (kk=0; kk<nn; ++kk) {
    xv = xx[kk] - xoff;
    ff = cpolyval(order+1,coef,xv);
    fprintf(outfu," %9.3f %9.3f %9.3f %9.3f \n",xx[kk],yy[kk],ff,yy[kk]-ff);
  }
  fclose(outfu);


  for (icol=0; icol<ecol; ++icol) {
    pixno = icol + (irow * nc);
    xv = (double)icol - xoff;
    offimg[pixno] = cpolyval(order+1,coef,xv);
  }

}

sprintf(wrd,"offimg_%d.fits",nso);
nsx_write_general_image( wrd, offimg,  nc, 200 );

}}
free(offimg);
return;
}



/* ----------------------------------------------------------------------
  Define object and background extraction windows in arcseconds.
  This loads the SPX[0].asp1,2 and SPX[0].nbk,abk1,2 values in arcseconds using
  asp1,2 or xsp1,xsp2, and nabk,abk1,2 or nxbk,xbk1,2 values..
  If noback=1, then do not set any background ranges.
  (User may request no background subtraction.)
  Returns '1' if no object window specified.
*/
int nsx_set_extraction_windows( AVPtype AVP[], SPXtype SPX[], 
                        double asp1, double asp2, double xsp1, double xsp2,
                        int nabk, double abk1[], double abk2[], 
                        int nxbk, double xbk1[], double xbk2[], int noback, IMGtype IMG )
{
/**/
int ii,NoWindow;
/**/

/* Object extraction window (based on nso=3 at column 1000). */
NoWindow = 0;
if ((asp1 > -9.)&&(asp2 > -9.)) {
  SPX[0].nsp     = 1;
  SPX[0].asp1[0] = asp1;
  SPX[0].asp2[0] = asp2;
  SPX[0].pflx[0] = 0.;
  SPX[0].sigs[0] = 0.;
  SPX[0].peak[0] = 0.;
} else {
  if ((xsp1 > -9.)&&(xsp2 > xsp1)) {
    SPX[0].nsp     = 1;
    SPX[0].asp1[0] = nsx_AVP( AVP, 3, 1000, xsp1, IMG );
    SPX[0].asp2[0] = nsx_AVP( AVP, 3, 1000, xsp2, IMG );
    SPX[0].pflx[0] = 0.;
    SPX[0].sigs[0] = 0.;
    SPX[0].peak[0] = 0.;
  } else {
    NoWindow = 1;
  }
}

/* Background extraction windows (based on nso=3 at column 1000). */
SPX[0].nbk=0;
if (noback == 0) {
  if (nabk > 0) {
    SPX[0].nbk = nabk;
    for (ii=0; ii<nabk; ++ii) {
      SPX[0].abk1[ii] = abk1[ii];
      SPX[0].abk2[ii] = abk2[ii];
    }
  } else {
    if (nxbk > 0) {
      SPX[0].nbk = nxbk;
      for (ii=0; ii<nxbk; ++ii) {
        SPX[0].abk1[ii] = nsx_AVP( AVP, 3, 1000, xbk1[ii], IMG );
        SPX[0].abk2[ii] = nsx_AVP( AVP, 3, 1000, xbk2[ii], IMG );
      }
    }
  }
}

return(NoWindow);
}


/* ----------------------------------------------------------------------
 Special wave correct routine.  -tab 30jan2018
*/
void nsx_wave_correct()
{
/**/
char roots[9][100];
char wrd[100];
char line[100];
/**/
int ii,nn,ff,nso,narr,kk,order,maxiter;
/**/
double rcol[9][3000];
double dpix[9][3000];
double arr[3000];
double dif[9],coef[9],xoff;
double xx[3000],yy[3000],ww[3000];
/**/
FILE *infu;
FILE *outfu;
/**/

printf("..............wave correct\n");
printf("..............wave correct\n");
printf("..............wave correct\n");

strcpy(roots[0],"s171102_0068");
strcpy(roots[1],"s171102_0070");
strcpy(roots[2],"s171102_0071");
strcpy(roots[3],"s171102_0074");

for (nso=3; nso<=7; ++nso) {

  nn=0;
  for (ff=0; ff<4; ++ff) {
    sprintf(wrd,"%s-pix%d.dat",roots[ff],nso);
    infu = fopen_read(wrd);
    nn=0;
    while (fgetline(line,infu)) {
      rcol[ff][nn] = GLV(line,1);
      dpix[ff][nn] = GLV(line,2);
      ++nn;
    }
    fclose(infu);
  }

  for (ff=0; ff<3; ++ff) {
    narr=0;
    for (ii=0; ii<nn; ++ii) {
      if (ABS((rcol[ff][ii] - rcol[3][ii])) > 0.000001) { 
        printf("***error: rcol mismatch(ff=%d,ii=%d) %f %f\n",ff,ii,rcol[ff][ii],rcol[3][ii]); 
        exit(1); 
      }
      arr[narr] = dpix[ff][ii] - dpix[3][ii];
      ++narr;
    }
    dif[ff] = cfind_median8(narr,arr);
    printf("nso=%d ff=%d  dif=%f\n",nso,ff,dif[ff]);
  }
  dif[3]=0.;

  kk=0;
  for (ff=0; ff<4; ++ff) {
    sprintf(wrd,"%s-pix%d.out",roots[ff],nso);
    outfu = fopen_write(wrd);
    for (ii=0; ii<nn; ++ii) {
      fprintf(outfu,"%f %f\n",rcol[ff][ii],dpix[ff][ii] - dif[ff]);
      xx[kk] = rcol[ff][ii];
      yy[kk] = dpix[ff][ii] - dif[ff];
      ww[kk] = 1.0;
      ++kk;
    }
    fclose(outfu);
  }

/* Fit. */
  order = 2;
  maxiter = kk/7;
  nsx_fitpoly_reject(kk,xx,yy,ww,order,maxiter,6.,&xoff,coef,1);
  sprintf(wrd,"wave_correct_B%d.dat",nso);

/*
  printf("%20.12e %20.12e %20.12e %20.12e\n",xoff,coef[0],coef[1],coef[2]);
  xv  = 1000. - xoff;
  tpix = cpolyval(order+1,coef,xv);
  printf("1000: tpix=%f\n",tpix);
  cpauseit();
*/

  outfu = fopen_write(wrd);
  fprintf(outfu," %20.12e %20.12e %20.12e %20.12e %20.12e \n",xoff,coef[0],coef[1],coef[2],coef[3]);
  fclose(outfu);
  cpauseit();

}

return;
}











/* ----------------------------------------------------------------------
 Fit a gaussian by using natural log and substition.
 
 Form:  y = (peak) * exp( -ln(2) * ((x / hwhm)^2) ).
 
 Change to:  Y = A X + D, where:
      Y == ln(y),  X == x^2,   A == -ln(2)/(hwhm^2),  D == ln(peak).
 
  Input: nn       : number of points.
         x8[]     : x values (will be altered).
         y8[]     : y values (will be altered).
         w8[]     : weight values.
 Output: peak     : Peak value.
         hwhm     : HWHM value.
*/
/*@@*/
void nsx_fit_gaussian( int nn, double x8[], double y8[], double w8[], double *peak, double *hwhm )
{
/**/
double coef[9];
/**/
int ii;
/**/
const double tiny = 1.e-50;
const double ee = 2.7182818284590452;
const double ln2 = 0.693147181;         /* natural logarithm of 2. */
/**/

/* Check. */
for (ii=0; ii<nn; ++ii) { if (y8[ii] < tiny) w8[ii]=0.; }

/* Load. */
for (ii=0; ii<nn; ++ii) {
  if ((w8[ii] > 0.)&&(y8[ii] > 0.)) {
/* Assume weight is s2n squared of y. */
    x8[ii] = x8[ii] * x8[ii];
    y8[ii] = log( y8[ii] );
  } else { w8[ii] = 0.; }
}
fit_straight_line( nn, x8, y8, w8, coef );

/* Translate. */
*peak=0.;  *hwhm=-1.;
if (coef[1] < 0.) {
  *peak = pow(ee,coef[0]);
  *hwhm = sqrt((  -1. * ln2 / coef[1] ));
}

/* Fit points. -- doesn't work since arrays are altered.
for (ii=0; ii<nn; ++ii) {
  rr = ABS(( x8[ii] / *hwhm ));
  if (rr < 4.) {
    r8 = -1. * ln2 * rr * rr;
    f8[ii] = *peak * exp(( r8 ));
  } else {
    f8[ii] = 0.;
  }
}
*/

return;
}




/* ----------------------------------------------------------------------
  Convert data to histogram data.
  Input:  nn        : number of data points
          yy[]      : data points
 Output:  nb        : number of bins
          xb        : x value of each bin
          yb        : y value of each bin
  Input:  max_nb    : maximum number of bins
 In/Out:  umin      : minimum bin value
          umax      : maximum bin value
          uinc      : (size of bins), xtick (see cpgbox()), nxtick (see cpgbox()),
 Output:  peak      : highest y bin value.
          fwhm      : Full Width Half Maximum estimate (0.939437 * area / peak).
          hi_yb     : highest y bin value (after possible log(y)).
  Input:  logy      : Convert y axis in binned data to log(y).
          normalize : Normalize bin data.
          outfile   : output file for bin data.
*/
/*@@*/
void nsx_Bin_Data( int nn, float yy[], int *nb, float xb[], float yb[], int max_nb,
                   float *umin, float *umax, float *uinc, float *peak, float *fwhm, float *hi_yb,
                   int logy, int normalize, char outfile[] )
{ 
/**/ 
float ylo,yhi,sum;
int ii,binnum;
/**/ 
FILE *outfu;
/**/

/* High and low of data. */
yhi = yy[0];
ylo = yy[0];
for (ii=0; ii<nn; ++ii) {
  if (yy[ii] < ylo) ylo = yy[ii];
  if (yy[ii] > yhi) yhi = yy[ii];
}

/* Decide on range and inc if needed. */
if (*umin < -1.e+29) { *umin = ylo; }
if (*umax >  1.e+29) { *umax = yhi; }
if (*uinc < -1.e+29) { *uinc = (*umax - *umin) / 100.; }

/* Number of bins. */
*nb = ((*umax - *umin) / *uinc) + 1.5;
if (*nb > max_nb) {
  fprintf(stderr,"***ERROR: Too many bins (%d), max is '%d'.\n",*nb,max_nb);
  exit(1);
}
printf("Bin_Data: ylo=%f  yhi=%f  nb=%d   umin=%f  umax=%f  uinc=%f \n",ylo,yhi,*nb,*umin,*umax,*uinc);

/* Initialize bins. */ 
for (ii=0; ii<*nb; ++ii) {
  xb[ii] = *umin + ( (*uinc * ii) + (*uinc/2.) );   /* middle of bin */
  yb[ii] = 0.;
}

/* Count number in bins (actual binning here). */
for (ii=0; ii<nn; ++ii) {
  binnum     = ( ( (yy[ii] - *umin) - (*uinc / 2.) ) / *uinc ) + 0.5;
  yb[binnum] = yb[binnum] + 1.;
} 

/* Normalize. */
if ((normalize == 1)&&(*nb > 0)) {
  for (ii=0; ii<*nb; ++ii) {
    yb[ii] = yb[ii] / ( (float)nn );
  }
}

/* FWHM estimate. */
sum   = 0.;
*peak = 0.;
for (ii=0; ii<*nb; ++ii) {
  sum = sum + yb[ii];
  if (yb[ii] > *peak) *peak = yb[ii];
}
if ((*peak > 0.)&&(sum > 0.)) {
  *fwhm = (0.939437 * (sum * (*uinc))) / *peak;
} else {
  *fwhm = -1.;
}

/* Make y axis logarithmic. */
if (logy == 1) {
  for (ii=0; ii<*nb; ++ii) {
    if (yb[ii] > 0.) { yb[ii] = log10(( yb[ii] )); } else { yb[ii] = -0.1; }
  }
}

/* Highest bin. */
*hi_yb = 0.;
for (ii=0; ii<*nb; ++ii) {
  if (yb[ii] > *hi_yb) { *hi_yb = yb[ii]; }
}

/* Write data? */
if (strcmp(outfile,"") != 0) {
  printf("NOTE: Writing histogram data to: %s .\n",outfile);
  outfu = fopen_write( outfile );
  for (ii=0; ii<*nb; ++ii) {
    fprintf(outfu,"%12.5e %12.5e\n",xb[ii],yb[ii]);
  }
  fclose(outfu);
}

return;
}



/* ----------------------------------------------------------------------
  Find hot pixels by comparing two dark images.
*/
void nsx_find_hot_pixels( IMGtype IA, IMGtype IB, SOPtype SOP1[], SOPtype SOP2[] )
{
/**/
int narr,pixno,nso,ecol,icol,irow,ii,jj,irow0;
/**/
double dif,ave,sigs,rowoff,cb1,cb2,IA_level,IB_level,edge;
double imgsumA,imgsumB,arrA[900],arrB[900];
/**/
float *imageA,*imageB,*imageHP;
float lim;
/**/
char wrd[100];
/**/

imageA = (float *)calloc(((IA.nc*IA.nr)+1000),sizeof(float));
imageB = (float *)calloc(((IA.nc*IA.nr)+1000),sizeof(float));
imageHP= (float *)calloc(((IA.nc*IA.nr)+1000),sizeof(float));

for (ii=0; ii<IA.nc; ++ii) {
for (jj=0; jj<IA.nr; ++jj) {
  pixno = ii + (jj * IA.nc);
  imageA[pixno] = 0.;
  imageB[pixno] = 0.;
  imageHP[pixno]= 0.;
}}

for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IA.nc/2; } else { ecol=IA.nc; }
  for (icol=0; icol<ecol; ++icol) {
    edge = nsx_find_real_image_row( 1, icol, nso );
    irow0= cnint(( edge ));

/* Level for this (slanted) column in IA and IB. */
    narr=0;
    for (irow=irow0; irow<(irow0+120); irow=irow+1) {
      rowoff = (double)irow - edge;
      nsx_slant_boundaries_SOP2( nso, icol, rowoff, SOP1, SOP2, ecol, &cb1, &cb2 );
      nsx_fractional_pixel_cb( IA.nc, IA.image, cb1, cb2, irow, &imgsumA );
      nsx_fractional_pixel_cb( IB.nc, IB.image, cb1, cb2, irow, &imgsumB );
      arrA[narr] = imgsumA;
      arrB[narr] = imgsumB;
      ++narr;
    }
    IA_level = cfind_median8(narr,arrA);
    IB_level = cfind_median8(narr,arrB);

/* Adjust two images. */
    for (irow=irow0; irow<(irow0+120); irow=irow+1) {
      rowoff = (double)irow - edge;
      nsx_slant_boundaries_SOP2( nso, icol, rowoff, SOP1, SOP2, ecol, &cb1, &cb2 );
      nsx_fractional_pixel_cb( IA.nc, IA.image, cb1, cb2, irow, &imgsumA );  imgsumA = imgsumA - IA_level;
      nsx_fractional_pixel_cb( IB.nc, IB.image, cb1, cb2, irow, &imgsumB );  imgsumB = imgsumB - IB_level;
      ii = cnint(( (cb1 + cb2)/2. ));
      pixno = ii + (irow * IA.nc);
      imageA[pixno] = imgsumA;
      imageB[pixno] = imgsumB;
    }

  }
}
    
sprintf(wrd,"imageA.fits"); nsx_write_general_image( wrd, imageA, IA.nc, IA.nr );
sprintf(wrd,"imageB.fits"); nsx_write_general_image( wrd, imageB, IA.nc, IA.nr );

/* Compare images, look for hot pixels. */
for (nso=3; nso<=7; ++nso) {
  lim=30.;
  if (nso == 3) lim=200.;
  if (nso == 4) lim=60.;
  if (nso == 7) { ecol=IA.nc/2; } else { ecol=IA.nc; }
/* NOTE: starting at column 9.. */
  for (icol=9; icol<ecol; ++icol) {
    edge = nsx_find_real_image_row( 1, icol, nso );
    irow0= cnint(( edge ));
    for (irow=irow0; irow<(irow0+120); irow=irow+1) {
      pixno = icol + (irow * IA.nc);
      if ((imageA[pixno] > lim)&&(imageB[pixno] > lim)) {
        ave = (imageA[pixno] + imageB[pixno]) / 2.;
        dif = ABS((imageA[pixno] - imageB[pixno]));
        sigs= dif / sqrt(ave);
        if (sigs < 3.) { imageHP[pixno] = ave; }
      }
    }
  }
}
      
sprintf(wrd,"imageHP.fits"); nsx_write_general_image( wrd, imageHP, IA.nc, IA.nr );


return;
}




/* ----------------------------------------------------------------------
 Used by gser() and gfc().
*/
/*@@*/
double gammln(double xx)
{
  double cof[6] = { 76.18009173, -86.50532033,     24.01409822,
                   -1.231739516,   0.120858003e-2, -0.536382e-5 };
  double stp = 2.50662827465 ;
  double half= 0.5;
  double one = 1.0;
  double fpf = 5.5;
  double x,tmp,ser;
  int j;
/**/
  x  = xx - one ;
  tmp= x + fpf ;
  tmp= ( (x+half) * log(tmp) ) - tmp ;
  ser= one ;
  for (j=0; j<=5; ++j) {
    x  = x + one ;
    ser= ser + (cof[j]/x) ;
  }
  return( tmp + log(stp*ser) );
}

/* ----------------------------------------------------------------------
  Returns an array of Poisson distribution points.
    Input:  mean     : expected mean count value.
            xx_start : First x value (usually 0.0).
            xx_end   : Last x value (e.g. 20.).
            xx_inc   : Increment for x values (warning: if not 1.0,
                         then the sum of distribution will not be 1.0)
                       (Also, sum will not be 1.0 if xx_start > 0 and/or
                        xx_end < infinity).
            maxarr   : Maximum points in array[].
   Output:  array[]  : Values as defined by inputs.
            nn       : Number of points in array.
  Note: The factorial can be expressed as a gamma function:
           z! == Gamma(z + 1) == integ(0 to inf){ exp(-t) * t^(z) dt }
*/
/*@@*/
void PoissonArray( double mean, double xx_start, double xx_end, double xx_inc,
                   int maxarr, double array[], int *nn )
{
/**/
double xx,xfac;
int kk;
/**/
kk=0;
for (xx=xx_start; xx<=xx_end; xx=xx+xx_inc) {
  xfac = 1.;
  if (xx > 0.) { xfac = exp(( gammln( xx + 1. ) )); }
  array[kk] = ( exp((-1. * mean)) * pow(mean,xx) ) / xfac;
  ++kk;
  if (kk >= maxarr) {
    fprintf(stderr,"***ERROR: PA: Exceeded array size [%d].\n",maxarr);
    exit(1);
  }
}
*nn = kk;
return;
}




/* ----------------------------------------------------------------------
  Given an area value (e.g. a random value between 0.0 and 1.0) compute x,
    where x is between -inf and inf (actually -5 to 5).  Sigma is 1.0.
 Input: np     : number of points in arrays.
        xx[]   : array of x points (area between 0 and this value).
        aa[]   : array of areas between 0 and xx[].
  Initialize random number generator using:  srand48(seed)
  Returns the x value.
*/
/*@@*/
float math_random_x( int np, float xx[], float aa[] )
{
/**/
float area,xv;
/**/
area = drand48();
xv = xx[(( cneari_bs4( area, np, aa ) ))];
if (drand48() < 0.5) { xv = -1. * xv; }
return(xv);
}




/* ----------------------------------------------------------------------
 For this gaussian, sigma is x=1.0 and the integral -inf to inf is 1.0 .
 Return gaussian value given "x".
*/
/*@@*/
double gaussian_sig( double x )
{
const double c = 0.398942280;   /*  c = 1 / sqrt(2*pi) */
return( c * exp((x*x)/(-2.)) );
}


/* ----------------------------------------------------------------------
  Integrate a gaussian and fill in an area function between 0 and ...
  Using:  y = c * exp(x^2/-2), where c= 2 * 1/sqrt(2*pi).
   Output: np     : number of points.
           xx[]   : x point (area between 0 and this value).
           aa[]   : area between 0 and xx[].
  Loads a 50,000 element array from x=0.0 to x=5.0.
*/
/*@@*/
void math_gaussian_area_array( int *np, float xx[], float aa[] )
{
/**/
double area,xai,sx,xinc,xv;
int ii,nn;
/**/
xx[0]= 0.;
aa[0]= 0.;
nn   = 1;
area = 0.;
xai  = 0.0001;
xinc= xai / 10.;
for (xv=0.; xv<5.; xv=xv+xai) {
  sx = xv;
  for (ii=0; ii<10; ++ii) {
    area = area + (gaussian_sig((sx + (xinc/2.))) * xinc);
    sx = sx + xinc;
  }
  xx[nn] = xv + (xai/2.);
  aa[nn] = 2. * area;
  ++nn;
}
*np = nn;
return;
}


/* ----------------------------------------------------------------------
 Return a noisy flux.  Call srand48(seed) and math_gaussian_area_array()
 before calling for the first time.
   Input:  flux   : flux.
           ferr   : 1 sigma error.
           ng     : from math_gaussian_area_array() routine.
           xg     : from math_gaussian_area_array() routine.
           ag     : from math_gaussian_area_array() routine.
*/
/*@@*/
float misc_FluxNoise( float flux, float ferr, int ng, float xg[], float ag[] )
{
/**/
float fnoise;
/**/
if (ferr > 0.) {
  fnoise = flux + ( (math_random_x( ng, xg, ag )) * ferr );
} else {
  fnoise = 0.;
}
return(fnoise);
}











/* ----------------------------------------------------------------------
  Compute variance given two flat field exposures of the same duration.. -tab 13feb2018

   var(e) = mean(e) = DN * eperdn
   DN = e / eperdn
   var(DN) = ( d DN / d e )^2  * var(e)  ..  propagation of errors
   var(DN) = ( ( 1 / eperdn )^2 ) * mean(e) 
   var(DN) = DN / eperdn
..
..
   D = a / b   , where a and b are flats of the same level..
   var(D) = ((d D / d a)^2) * var(a)   +  ((d D / d b)^2) * var(b)   .. propagation of errors
          = ( 1 / b )^2  * var(a)  +  ( a / b )^2 * var(b)

  Run with(e.g.): nsx s170216_0054.fits s170216_0055.fits eperdn=1.0 boxrad=20
*/
void nsx_compute_variance2( IMGtype IA, IMGtype IB, double eperdn, int boxrad )
{
/**/
int pixno,ecol,ii,jj,icol,narr,irow,iedge,nso,nc,nr;
int ii1,ii2,jj1,jj2;
/**/
char wrd[100];
/**/
double rms,median,Asum,Bsum,arr[9900];
double mean,Amean,Bmean,pois,err;
/**/
float *simimg;
float *divimg;
float *subimg;
float *sdivimg;
float *ssubimg;
float *scrimg;
/**/
FILE *outfu;
FILE *outfu2;
/**/
const int max_nb = 9000;
int nb,logy,normalize;
/**/
const int max_yy = 32000;
float *yy;
float xb[max_nb],yb[max_nb];
float umin,umax,uinc,peak,fwhm,hi_yb;
/**/
const int max8 = 200;
double x8[max8],y8[max8],w8[max8];
double rr,r8,peak8,hwhm8;
const double ln2 = 0.693147181;         /* natural logarithm of 2. */
int gg,nn;
double limit;
/**/
int ng;
float xg[51000],ag[51000];
double gflux,gerror;
/**/


/* Allocate. */
yy = (float *)calloc(max_yy,sizeof(float));

/* Echo. */
printf("file A is '%s' exp=%f\n",IA.file,IA.exptime);
printf("file B is '%s' exp=%f\n",IB.file,IA.exptime);
printf("eperdn=%f   boxrad=%d \n",eperdn,boxrad);
nc=IA.nc; nr=IA.nr;

/* Allocate. */
simimg = (float *)calloc(((nc*nr)+1000),sizeof(float));
divimg = (float *)calloc(((nc*nr)+1000),sizeof(float));
subimg = (float *)calloc(((nc*nr)+1000),sizeof(float));
sdivimg= (float *)calloc(((nc*nr)+1000),sizeof(float));
ssubimg= (float *)calloc(((nc*nr)+1000),sizeof(float));
scrimg = (float *)calloc(((nc*nr)+1000),sizeof(float));
for (pixno=0; pixno<(nc*nr); ++pixno) { 
  divimg[pixno] =0.; 
  subimg[pixno] =0.; 
  sdivimg[pixno]=0.; 
  ssubimg[pixno]=0.; 
  scrimg[pixno] =0.; 
}

/* Only 2 orders. */
for (nso=5; nso<=6; ++nso) { if (nso > 0) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (icol=0; icol<ecol; ++icol) {
    iedge = cnint(( nsx_find_real_image_row( 1, icol, nso ) ));
    for (irow=iedge; irow<(iedge+110); irow=irow+1) {
      pixno = icol + (irow * nc);
      if (IB.image[pixno] > 0.) {
        divimg[pixno] = IA.image[pixno] / IB.image[pixno];
        subimg[pixno] = IA.image[pixno] - IB.image[pixno];
      }
    }
  }
}}

/* Smooth the images. */
for (pixno=0; pixno<(nc*nr); ++pixno) { sdivimg[pixno]=divimg[pixno]; }
for (pixno=0; pixno<(nc*nr); ++pixno) { ssubimg[pixno]=subimg[pixno]; }
for (pixno=0; pixno<(nc*nr); ++pixno) { simimg[pixno] =IA.image[pixno]; }
if (boxrad < 80) {
  misc_Smooth_Image( nc, nr, sdivimg, scrimg, boxrad, boxrad );
  misc_Smooth_Image( nc, nr, ssubimg, scrimg, boxrad, boxrad );
  misc_Smooth_Image( nc, nr, simimg,  scrimg, boxrad, boxrad );   /* For noise simulation test.. */
} else {
  printf("no smoothing..\n");
}


/* Load gaussian area array. */
math_gaussian_area_array( &ng, xg, ag );   

sprintf(wrd,"flat-smt.fits"); nsx_write_general_image( wrd, simimg,  nc, nr );

/* Make noisey.. */
if (pixno == -43244) {
for (pixno=0; pixno<(nc*nr); ++pixno) { 
  if (simimg[pixno] > 1000.) {
    simimg[pixno] = 1000.;
    gflux = simimg[pixno];
    gerror= sqrt((gflux));    /* 1 sigma error */
    simimg[pixno] = misc_FluxNoise( gflux, gerror, ng,xg,ag );
  }
}
sprintf(wrd,"flat-sim.fits"); nsx_write_general_image( wrd, simimg,  nc, nr );
}


sprintf(wrd,"flat-div.fits"); nsx_write_general_image( wrd, divimg,  nc, nr );
sprintf(wrd,"flat-sub.fits"); nsx_write_general_image( wrd, subimg,  nc, nr );
sprintf(wrd,"flat-sdiv.fits"); nsx_write_general_image( wrd, sdivimg,  nc, nr );
sprintf(wrd,"flat-ssub.fits"); nsx_write_general_image( wrd, ssubimg,  nc, nr );

/* Correct sub image.. */
if (boxrad < 80) {
  for (pixno=0; pixno<(nc*nr); ++pixno) { 
    subimg[pixno] = subimg[pixno] - ssubimg[pixno]; 
  }
}
sprintf(wrd,"flat-csub.fits"); nsx_write_general_image( wrd, subimg,  nc, nr );


/* Poission from subtraction ... Only 2 orders. */

sprintf(wrd,"limit_b%2.2d_e%0.2f.tbl",boxrad,eperdn);
outfu2 = fopen_write(wrd);
fprintf(outfu2,"| limit  | peak8   | hwhm8   | pois    | nn  | H_fwhm  | sqrt    | sigma   |\n");

for (gg=1; gg<40; ++gg) {

  limit  = (double)gg * 500.; 
  printf("gg=%4d  limit=%f\n",gg,limit);

/* Load yy[] .. */
  nn=0;
  for (nso=5; nso<=6; ++nso) {
    if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
    for (icol=10; icol<ecol-10; icol=icol+16) {
      iedge = cnint(( nsx_find_real_image_row( 1, icol, nso ) ));
      for (irow=iedge+10; irow<(iedge+100); irow=irow+16) {
        ii1 = icol-8;
        ii2 = icol+8;
        jj1 = irow-8;
        jj2 = irow+8;
        narr=0; Asum=0.; Bsum=0.;
        for (ii=ii1; ii<=ii2; ++ii) {
        for (jj=jj1; jj<=jj2; ++jj) {
          pixno = ii + (jj * nc);
          if ((IA.image[pixno] > 0.)&&(subimg[pixno] > -1000.)&&(subimg[pixno] < 1000.)) {
            Asum = Asum + IA.image[pixno];
            Bsum = Bsum + IB.image[pixno];
            arr[narr] = subimg[pixno];
            ++narr;
          }
        }
        }
        if (narr < 0) narr=1;
        median = cfind_median8(narr,arr);
        rms = 0.;
        for (ii=0; ii<narr; ++ii) {
          rms = rms + ( (arr[ii] - median) * (arr[ii] - median) );
        }
        rms = sqrt(( rms / (double)narr ));
        Amean = Asum / (double)narr;
        Bmean = Bsum / (double)narr;
        mean= (Amean + Bmean) / 2.;
        if ((mean > (limit-100.))&&(mean < (limit+100.))) {
          for (ii=0; ii<narr; ++ii) {
            yy[nn] = ABS(( arr[ii] - median ));
            ++nn;
            if (nn > max_yy-3) { printf("***error: too many yy[] entries.\n"); exit(1); }
          }
        }
      }    /* for (irow=iedge+10; irow<(iedge+100); irow=irow+16) .. */
    }      /* for (icol=10; icol<ecol-10; icol=icol+16) .. */
  }        /* for (nso=5; nso<=6; ++nso) .. */

/* Fit guassian. */
  umin=0.; umax=+300.; uinc=10.; logy=0; normalize=0;
  nsx_Bin_Data( nn, yy, &nb, xb, yb, max_nb, &umin, &umax, &uinc, &peak, &fwhm, &hi_yb, logy, normalize, "jbin.dat" );
  printf("Bin_Data: limit=%8.1f  nn=%d  peak=%f  fwhm=%f  hi_yb=%f  \n",limit,nn,peak,fwhm,hi_yb);
  if (nb > max8-3) { printf("***error: too many bins for max8..\n"); exit(1); }
  for (ii=0; ii<nb; ++ii) {
    x8[ii] = xb[ii];
    y8[ii] = yb[ii];
    w8[ii] = 1.0;
    rr = x8[ii]; 
    if (rr < 0.2*fwhm) rr=0.2*fwhm;
    if (rr > 3.*fwhm) { w8[ii]=0.; } else { w8[ii] = fwhm / rr; }
  }
  nsx_fit_gaussian( nb, x8, y8, w8, &peak8, &hwhm8 );
  pois = sqrt(( limit / eperdn )) * 1.17741;
  printf("Fit_Gaussian: mean=%8.1f   peak8=%9.2f  hwhm8=%9.4f  pois=%9.4f\n",limit,peak8,hwhm8,pois);

  fprintf(outfu2," %8.1f %9.2f %9.4f %9.4f %5d %9.4f %9.4f %9.4f \n",
    limit,peak8,hwhm8,pois,nn,fwhm,sqrt((limit/eperdn)),sqrt((limit/eperdn)));

/* Reload, compute, show. */
  for (ii=0; ii<nb; ++ii) {
    rr = ABS(( (double)xb[ii] / hwhm8 ));
    if (rr < 4.) {
      r8 = -1. * ln2 * rr * rr;
      w8[ii] = peak8 * exp(( r8 ));
    } else {
      w8[ii] = 0.;
    }
  }

/*
  sprintf(wrd,"jg%5.5d.tbl",cnint(limit));
  printf("writing '%s'\n",wrd);
  outfu = fopen_write(wrd);
  fprintf(outfu,"| xx         | yy         | ff         |\n");
  for (ii=0; ii<nb; ++ii) {
    fprintf(outfu," %12.6f %12.6f %12.6f \n",xb[ii],yb[ii],w8[ii]);
  }
  fclose(outfu);
*/

}

fclose(outfu2);



if (nso != 32345) return;


/* Poission from flat division... Only 2 orders. */
outfu = fopen_write("pois_div.tbl");
fprintf(outfu,"|nso|icol| mean       | err     | pois    |\n");
for (nso=5; nso<=6; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (icol=10; icol<ecol-10; icol=icol+16) {
    iedge = cnint(( nsx_find_real_image_row( 1, icol, nso ) ));
    for (irow=iedge+10; irow<(iedge+100); irow=irow+16) {
      ii1 = icol-8;
      ii2 = icol+8;
      jj1 = irow-8;
      jj2 = irow+8;
      narr=0; Asum=0.; Bsum=0.;
      for (ii=ii1; ii<=ii2; ++ii) {
      for (jj=jj1; jj<=jj2; ++jj) {
        pixno = ii + (jj * nc);
        if ((IA.image[pixno] > 0.)&&(divimg[pixno] > 0.7)&&(divimg[pixno] < 1.3)) {
          Asum = Asum + IA.image[pixno];
          Bsum = Bsum + IB.image[pixno];
          arr[narr] = divimg[pixno];
          ++narr;
        }
      }
      }
      if (narr < 0) narr=1;
      median = cfind_median8(narr,arr);
      rms = 0.;
      for (ii=0; ii<narr; ++ii) {
        rms = rms + ( (arr[ii] - median) * (arr[ii] - median) );
      }
      rms = sqrt(( rms / (double)narr ));
      Amean = Asum / (double)narr;
      Bmean = Bsum / (double)narr;
      mean= (Amean + Bmean) / 2.;
      err = (rms * mean) * sqrt(( 0.5 ));
      pois= sqrt(( mean/eperdn ));
      fprintf(outfu," %3d %4d %12.5f %9.5f %9.5f \n",nso,icol,mean,err,pois);
    }
  }
}
fclose(outfu);

return;
}



/* ----------------------------------------------------------------------
  Compute variance given two exposures of the same duration.. -tab 09feb2018
 (NOTE: This may not be quite right since the error is not the difference in
  the same pixel, but the deviation from the mean or 'true' value.. -tab 13feb2018)
*/
void nsx_compute_variance( IMGtype IA, IMGtype IB )
{
/**/
int nn,bb,pixno,nso,kk,ii,jj,jjoff;
/**/
char wrd[100];
/**/
FILE *outfu;
FILE *outfu2;
/**/
double num[1000],pois,ave,err;
const double eperdn = 0.5;
/**/
int narr,nss;
double median,arr[99000];
double x8[1000],y8[1000],w8[1000],peak,hwhm;
/**/
double ea[100][1000];
int nea[1000];
int ss,Snso[9],Sii1[9],Sii2[9];
/**/

printf("file A is '%s' exp=%f\n",IA.file,IA.exptime);
printf("file B is '%s' exp=%f\n",IB.file,IA.exptime);

Snso[0]=5; Sii1[0]= 800; Sii2[0]=880;
Snso[1]=6; Sii1[1]= 680; Sii2[1]=780;
Snso[2]=6; Sii1[2]=1450; Sii2[2]=1550;
nss = 3;

/* Sections. */
for (ss=0; ss<nss; ++ss) {
  sprintf(wrd,"tst%d.dat",ss);
  outfu = fopen_write(wrd);
  fprintf(outfu,"| ave        | err        | sqrt       |nso| ii  | jj  |\n");
  nso = Snso[ss];
  for (ii=Sii1[ss]; ii<=Sii2[ss]; ++ii) {
    jjoff  = cnint((nsx_find_real_image_row( 1, ii, nso )));
    for (jj=jjoff; jj<(jjoff+110); ++jj) {
      pixno = ii + (jj * IA.nc);
      ave = (IA.image[pixno] + IB.image[pixno]) / 2.;
      err = ABS(( IA.image[pixno] - IB.image[pixno] ));
      fprintf(outfu,"%12.5f %12.5f %12.5f %3d %5d %5d \n",ave,err,sqrt(ave*eperdn),nso,ii,jj);
    }
  }
  fclose(outfu);
}


/* General. */
for (ii=0; ii<1000; ++ii) { nea[ii]=0; }
printf("open to write tstg.tbl\n");
outfu = fopen_write("tstg.tbl");
fprintf(outfu,"| ave        | err        | sqrt       |nso| ii  | jj  |\n");
for (nso=5; nso<=6; ++nso) {
  for (ii=0; ii<IA.nc; ++ii) {
    jjoff  = cnint((nsx_find_real_image_row( 1, ii, nso )));
    for (jj=jjoff; jj<(jjoff+60); ++jj) {
      pixno = ii + (jj * IA.nc);
      ave = (IA.image[pixno] + IB.image[pixno]) / 2.;
      err = ABS(( IA.image[pixno] - IB.image[pixno] ));
      kk = cnint(( ave / 200. ));
      if ((kk < 100)&&(kk > 0)) {
        ea[kk][nea[kk]] = err;
        nea[kk] = nea[kk] + 1;
      }
      if ((ave > 1.)&&(ave < 22001.)) {
        fprintf(outfu,"%12.5f %12.5f %12.5f %3d %5d %5d \n",ave,err,sqrt(ave*eperdn),nso,ii,jj);
      }
    }
  }
}
fclose(outfu);



/*
outfu = fopen_write("tst.draw");
fprintf(outfu,"sci 3 ; sym 4\n");
for (ave=100.; ave<9000.; ave=ave+100.) {
  fprintf(outfu,"%f %f\n",ave,sqrt(eperdn*ave));
}
fprintf(outfu,"plot\n");
fclose(outfu);
*/


printf("open to write nea.tbl\n");
outfu = fopen_write("nea.tbl");
fprintf(outfu,"| kk  | median     | num     |\n");
for (kk=0; kk<100; ++kk) {
  narr=0;
  for (ii=0; ii<nea[kk]; ++ii) { arr[narr] = ea[kk][ii];  ++narr; }
  median = cfind_median8(narr,arr);
  fprintf(outfu," %5d %12.5f %9d \n",kk,median,nea[kk]);
  sprintf(wrd,"nea%3.3d.dat",kk);
  printf("wrd='%s'\n",wrd);
  outfu2 = fopen_write(wrd);
  fprintf(outfu2,"| err        |\n");
  for (ii=0; ii<nea[kk]; ++ii) { 
    fprintf(outfu2," %12.7f \n",ea[kk][ii]); 
  }
  fclose(outfu2);
}
fclose(outfu);


printf("open to write res.tbl\n");
outfu2 = fopen_write("res.tbl");
fprintf(outfu2,"| ave     | pois    | hwhm    |\n");
for (kk=10; kk<90; ++kk) {
  sprintf(wrd,"eab%2.2d.tbl",kk);
  outfu = fopen_write(wrd);
  fprintf(outfu,"| err  | num     |\n");
  for (bb=0; bb<100.; ++bb) { num[bb]=0.; }
  for (ii=0; ii<nea[kk]; ++ii) { 
    bb = cnint(( ea[kk][ii] / 10. ));  
    num[bb] = num[bb] + 1.;
  }
  nn=0;
  for (bb=2; bb<100.; ++bb) { 
    fprintf(outfu," %6.1f %9.1f \n",(double)bb * 10., num[bb]);
    if (num[bb] > 5.) {
      x8[nn] = (double)bb * 10.;
      y8[nn] = num[bb];
      w8[nn] = 1.0;
      ++nn;
    }
  }
  fclose(outfu);
  nsx_fit_gaussian( nn, x8, y8, w8, &peak, &hwhm );
  ave = (double)kk * 200.;
  pois= sqrt((ave));
  printf("nea=%5d  kk=%5d  ave=%9.2f  pois=%9.3f  peak=%9.1f  hwhm=%9.3f \n",nea[kk],kk,ave,pois,peak,hwhm);
  fprintf(outfu2," %9.3f %9.3f %9.3f \n",ave,pois,hwhm);
}
fclose(outfu2);
  


return;
}


/* ----------------------------------------------------------------------
  Change ASPP in AVP calibration data files.  -tab 30jul2018
  Define new value (aspp_new) and also compute current existing value
  using loaded AVP[] and  nso=3,4,5,6 (not 7)  (see TestProCent())

  Note that the original value of 0.123 taken from NIRES web page and then
  used in original (pre July 2018) AVP..dat file applies the 0.123 as/px to
  nso=3, the bluer orders are more like 0.120(nso=4), 0.117(nso=5), and
  0.115(nso=6) .. Here aspp_new and aspp_old both use nso=3..6 ..
*/
void nsx_Change_ASPP_in_AVP( AVPtype AVP[], char nsxdir[], char UseAVP[] )
{
/**/
int nn,ecol,nso,col,orderavp,orderinv;
const int nc = nc_Nominal;    /* 2048 */
/**/
const int marr = 9000;
double arr[marr];
/**/
double average,median,sum,num,as1,as2,rowoff,aspp_old;
const double aspp_new = 0.150;
/**/
double xoffavp,coefavp[9];
double xoffinv,coefinv[9];
double high,wrms,wppd,xv,xx[200],yy[200],ww[200];
/**/
char line[200];
char wrd[200];
/**/
FILE *infu;
FILE *outfu;
FILE *outfu2;
/**/
IMGtype IMG;
/**/

/* For this routine, there is no DAR (Differential Atmospheric Refraction)
   correction, so set .el to zero. */
IMG.el      = 0.;
IMG.rotposn = 0.;
IMG.parang  = 0.;

/* First compute scaling factor using nso=3..6 . */
/* (See nsx_TestProCent() and 'tpc.info' which uses only nso=4.) */
outfu = fopen_write("ASPPinAVP.tbl");
fprintf(outfu,"|nso| col |rowoff | Defin  | aspp   | as1    | as2    |\n");
nn=0;
for (nso=3; nso<=6; ++nso) {
  sum=0.; num=0.;
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (col=0; col<ecol; col=col+50) {
    for (rowoff=10.; rowoff<110; rowoff=rowoff+5.) {
      as1 = nsx_AVP( AVP, nso, col, rowoff, IMG );
      as2 = nsx_AVP( AVP, nso, col, rowoff+1., IMG );
      arr[nn] = as2 - as1;
      sum = sum + arr[nn];
      num = num + 1.;
      fprintf(outfu," %3d %5d %7.2f %8.5f %8.5f %8.5f %8.5f \n",
         nso,col,rowoff,ARCSEC_PER_PIXEL,(as2-as1),as1,as2);
      ++nn;
      if (nn > marr-3) { printf("***error: too many ASPPinAVP values.\n"); exit(1); }
    }
  }
}
average = sum / num;
median = cfind_median8(nn,arr);
printf("Average ASPP is %12.7f [median=%12.7f] for nso=3..6 ..  aspp_new=%12.7f\n",average,median,aspp_new);
fclose(outfu);

/* Use median. */
aspp_old = median;
printf("Use aspp_old=%f\n",aspp_old);

printf("UseAVP='%s'\n",UseAVP);

/* Always use 2nd order. */
orderavp = 2;


/* Load AVP (arcsec vs. pixel) polnomials.. */
sprintf(wrd,"%scal/AVP.%s.dat",nsxdir,UseAVP); printf("Read '%s'.\n",wrd);
infu  = fopen_read(wrd);
sprintf(wrd,"%scal/AVP.%s.new",nsxdir,UseAVP); printf("Write '%s'.\n",wrd);
outfu = fopen_write(wrd);
sprintf(wrd,"%scal/AVPinv.%s.new",nsxdir,UseAVP); printf("Write '%s'.\n",wrd);
outfu2= fopen_write(wrd);
while (fgetline(line,infu)) {
  nso = GLV(line,1);
  col = GLV(line,2);
  AVP[nso].xoff[col]    = GLV(line,3);
  AVP[nso].coef[col][0] = GLV(line,4);
  AVP[nso].coef[col][1] = GLV(line,5);
  AVP[nso].coef[col][2] = GLV(line,6);
  nn=0;
  for (rowoff=0.; rowoff<116; rowoff=rowoff+1.) {
    xx[nn] = rowoff;
    xv = rowoff - AVP[nso].xoff[col];
    yy[nn] = cpolyval( orderavp+1, AVP[nso].coef[col], xv );
    yy[nn] = yy[nn] * (aspp_new / aspp_old);
    ww[nn] = 1.0;
    ++nn;
  }

  orderavp = 2;
  xoffavp = AVP[nso].xoff[col];
  if (GJ_polyfit(nn,xx,yy,ww,orderavp,1,&xoffavp,coefavp) != 1) { printf("***error: for fit failed:Change_ASPP.\n"); exit(1); }
  GJ_polyfit_residuals( nn, xx, yy, ww, orderavp, xoffavp, coefavp, &high, &wrms, &wppd );
  if (ABS((high)) > 0.0000001) {
    printf("for: nso=%d col=%4d : nn=%3d  high=%12.7f  wrms=%12.7f  wppd=%12.7f \n",nso,col,nn,high,wrms,wppd);
    cpauseit();
  }
  fprintf(outfu," %3d %4d %20.12e %20.12e %20.12e %20.12e \n",nso,col,xoffavp,coefavp[0],coefavp[1],coefavp[2]);

  orderinv = 2;
  if (GJ_polyfit(nn,yy,xx,ww,orderinv,0,&xoffinv,coefinv) != 1) { printf("***error: inv fit failed:Change_ASPP.\n"); exit(1); }
  GJ_polyfit_residuals( nn, yy, xx, ww, orderinv, xoffinv, coefinv, &high, &wrms, &wppd );
  if (col ==   0) { printf("col=  0 for nso=%d\n",nso); }
  if (col == 800) { printf("col=800 for nso=%d\n",nso); }
  if (ABS((high)) > 0.0150) {
    printf("BAD: inv: nso=%d col=%4d : nn=%3d  high=%12.7f  wrms=%12.7f  wppd=%12.7f \n",nso,col,nn,high,wrms,wppd);
    exit(1);
  } else {
    if (ABS((high)) > 0.0110) {
      printf(" OK: inv: nso=%d col=%4d : nn=%3d  high=%12.7f  wrms=%12.7f  wppd=%12.7f \n",nso,col,nn,high,wrms,wppd);
      cpauseit();
    }
  }
  fprintf(outfu2," %3d %4d %20.12e %20.12e %20.12e %20.12e \n",nso,col,xoffinv,coefinv[0],coefinv[1],coefinv[2]);

}
fclose(infu);
fclose(outfu);
fclose(outfu2);
  
return;
}



/* ----------------------------------------------------------------------
  Check the AVP ASPP values..   -tab 30jul2018
*/
void nsx_Check_AVP_ASPP( AVPtype AVP[] )
{
/**/
int ecol,nso,col;
const int nc = nc_Nominal;  /* 2048 */
/**/
double as1,as2,rowoff;
/**/
FILE *outfu;
/**/
IMGtype IMG;
/**/

/* For this routine, there is no DAR (Differential Atmospheric Refraction)
   correction, so set .el to zero. */
IMG.el      = 0.;
IMG.rotposn = 0.;
IMG.parang  = 0.;

printf("Writing aspp.tbl\n");
outfu = fopen_write("aspp.tbl");
fprintf(outfu,"|nso| col |rowoff | Defin  | aspp   | as1    | as2    |\n");
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (col=0; col<ecol; col=col+400) {
    for (rowoff=20.; rowoff<100; rowoff=rowoff+20.) {
      as1 = nsx_AVP( AVP, nso, col, rowoff, IMG );
      as2 = nsx_AVP( AVP, nso, col, rowoff+1., IMG );
      fprintf(outfu," %3d %5d %7.2f %8.5f %8.5f %8.5f %8.5f \n",
         nso,col,rowoff,ARCSEC_PER_PIXEL,(as2-as1),as1,as2);
    }
  }
}
fclose(outfu);

return;
}




/* ----------------------------------------------------------------------
  Change AVP polynomials.. -tab 18may2018    [ NewAVP ]
  ( for s180120_0056: TRACE_TEST_ARCSEC 7.1312 )
*/
void nsx_ChangeAVP( AVPtype AVP[], char nsxdir[] )
{
/**/
char line[300];
char root[40];
char wrd[200];
char utcode[40];
/**/
FILE *infu;
FILE *forfu;
FILE *invfu;
/**/
int nc,nn,orderinv,orderavp,jj,mw,col,ecol,nso,ii;
/**/
double xx[300],yy[300],ww[300];
double xoffavp,coefavp[9],xoffinv,coefinv[9];
double rowoff,high,wrms,wppd;
double edge,arcsec;
/**/
AVPtype AVP2[9];
/**/
FILE *datfu;
FILE *outfu;
FILE *infofu;
/**/
double as,asinc,asmax,sum,num,rb1,rb2,imgsum,cen_as;
int nexp,iexp,order,nnt,ii1,ii2,maxiter;
double ff,xofft,coeft[9],coef[9];
const int mmt = 800;
double xxt[mmt],yyt[mmt],wwt[mmt];
/**/
const int mcol = 2100;
const int mexp = 20;
double *cas[mexp][8];
char ffile[mexp][200];
double arcsec0[mexp];
/**/
SPXtype SPX[9];
IMGtype IA;
IMGtype IMG;
/**/
int NoClean = 0;
int NoHotClean = 0;
/**/

/* For this routine, there is no DAR (Differential Atmospheric Refraction)
   correction, so set .el to zero. */
IMG.el      = 0.;
IMG.rotposn = 0.;
IMG.parang  = 0.;

/* Set */
nsx_clear_AVP( AVP2 );
if (FileExist("change.info") == 0) { printf("***error: no change.info file.\n"); exit(1); }

/* Allocate. */
for (iexp=0; iexp<mexp; ++iexp) {
for (nso=3; nso<=7; ++nso) { cas[iexp][nso] = (double *)calloc(mcol,sizeof(double)); }}

/* Clear ChangeArcSecond array. */
for (iexp=0; iexp<mexp; ++iexp) {
for (nso=3; nso<=7; ++nso) {
for (col=0; col<mcol; ++col) { cas[iexp][nso][col] = 0.; }}}

/* Set UT code for new AVP files.  Save info file in cal directory. */
infofu=fopen_read("change.info"); fgetline(line,infofu); fclose(infofu);
nc = GLV(line,1);
ii = cindex(line,"UT=");
substrcpy_terminate(line,ii+3,ii+12,utcode,0);
printf(".....................ChangeAVP...................[%s]\n",utcode);
fprintf(logfu,".....................ChangeAVP...................[%s]\n",utcode);
sprintf(wrd,"cp change.info %scal/AVP.%s.info",nsxdir,utcode); misc_system(wrd);

/* New AVP files. */
sprintf(wrd,"%scal/AVP.%s.dat",nsxdir,utcode); 
printf("Writing '%s'.\n",wrd);
fprintf(logfu,"Writing '%s'.\n",wrd);
if (FileExist(wrd)) { printf("===warning: New AVP filename already exists.\n"); }
forfu = fopen_write(wrd);
sprintf(wrd,"%scal/AVPinv.%s.dat",nsxdir,utcode); 
printf("Writing '%s'.\n",wrd);
fprintf(logfu,"Writing '%s'.\n",wrd);
invfu = fopen_write(wrd);

/* Read all lines from info file. Load up cas[][][] array. */
infofu = fopen_read("change.info");
nexp=0;
while (fgetline(line,infofu)) { printf("%s\n",line);

  ii = GLV(line,1); if (ii != nc) { printf("***error: bad nc\n"); exit(1); }

  arcsec0[nexp] = GLV(line,2);

  ii = cindex(line,"file=");
  substrcpy_terminate(line,ii+5,clc(line),wrd,0);
  ii=cindex(wrd," "); if (ii > 0) wrd[ii]='\0';
  strcpy(ffile[nexp],wrd);

  ii = cindex(line,"root=");
  substrcpy_terminate(line,ii+5,clc(line),root,0);
  ii=cindex(root," "); if (ii > 0) root[ii]='\0';

  for (nso=3; nso<=7; ++nso) {
    sprintf(wrd,"%s.change%d",root,nso);
    printf("Read '%s'.\n",wrd);
    datfu = fopen_read(wrd);
    while (fgetline(line,datfu)) {
      col = GLV(line,1);
      cas[nexp][nso][col] = GLV(line,2);
    }
    fclose(datfu);
  }
  ++nexp;
  if (nexp > mexp-3) { printf("***error:(new AVP): too many exposures.\n"); exit(1); }
}
fclose(infofu);
printf("Read %d exposure data files.\n",nexp);
  

/* Each order. */
for (nso=3; nso<=7; ++nso) {  
   if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
   mw = nsx_minwidth(ecol,nso);

/* Each column in each echelle order. */
  for (col=0; col<ecol; ++col) {
/* Fit flat or straight line to points within a column. */
    nn=0;
    for (iexp=0; iexp<nexp; ++iexp) {
      xx[nn] = arcsec0[iexp];
      yy[nn] = cas[iexp][nso][col];
      ww[nn] = 1.0;
      ++nn;
    }
    if (nn == 0) { printf("***error: no points found.\n"); exit(1); }
/* Flat line (default). */
    sum=0.; for (ii=0; ii<nn; ++ii) { sum=sum+yy[ii]; }
    coef[0]= sum / (double)nn;
    order  = 0;
/* Straight line fit. */
    if (nn > 1) { 
      order = 1;
      fit_straight_line( nn, xx, yy, ww, coef );
    }


/*
     The 'change_as' values are a fit to 'cen_as - arcsec0'.  If the new position of a star is below
     the 'arcsec0' value, then we want that position (in rowoff space) to now be 'arcsec0'. So at
     the 'cen_as' position (in arcsec space), subtract cen_as and add arcsec0..
*/

/* Apply. */
    nn=0;
    for (jj=0; jj<=mw; ++jj) {
      rowoff = (double)jj;
      arcsec = nsx_AVP( AVP, nso, col, rowoff, IMG );
      xx[nn] = rowoff;
      yy[nn] = arcsec - cpolyval(order+1,coef,arcsec);
      ww[nn] = 1.0;
      ++nn;
    }

    orderavp = 2;
    if (GJ_polyfit(nn,xx,yy,ww,orderavp,0,&xoffavp,coefavp) != 1) { printf("***error: for fit failed:chanageavp.\n"); exit(1); }
    GJ_polyfit_residuals( nn, xx, yy, ww, orderavp, xoffavp, coefavp, &high, &wrms, &wppd );
    if (ABS((high)) > 0.01) {
      printf("for: nso=%d col=%4d : nn=%3d  high=%12.7f  wrms=%12.7f  wppd=%12.7f \n",nso,col,nn,high,wrms,wppd);
      exit(1);
    }

    orderinv = 2;
    if (GJ_polyfit(nn,yy,xx,ww,orderinv,0,&xoffinv,coefinv) != 1) { printf("***error: inv fit failed:changeavp.\n"); exit(1); }
    GJ_polyfit_residuals( nn, yy, xx, ww, orderinv, xoffinv, coefinv, &high, &wrms, &wppd );
    if (ABS((high)) > 0.015) {
      printf("inv: nso=%d col=%4d : nn=%3d  high=%12.7f  wrms=%12.7f  wppd=%12.7f \n",nso,col,nn,high,wrms,wppd);
      exit(1);
    }

/* Write out new AVP..dat and AVPinv..dat .. */
    fprintf(forfu," %3d %4d %20.12e %20.12e %20.12e %20.12e \n",nso,col,xoffavp,coefavp[0],coefavp[1],coefavp[2]);
    fprintf(invfu," %3d %4d %20.12e %20.12e %20.12e %20.12e \n",nso,col,xoffinv,coefinv[0],coefinv[1],coefinv[2]);

  }   /* for (col=0 .. */
}     /* for (nso=3 .. */

fclose(forfu);
fclose(invfu);


/* Load new AVP values. */
sprintf(wrd,"%scal/AVP.%s.dat",nsxdir,utcode);
printf("Read '%s'.\n",wrd);
infu = fopen_read(wrd);
while (fgetline(line,infu)) {
  nso = GLV(line,1);
  ii  = GLV(line,2);
  AVP2[nso].xoff[ii]    = GLV(line,3);
  AVP2[nso].coef[ii][0] = GLV(line,4);
  AVP2[nso].coef[ii][1] = GLV(line,5);
  AVP2[nso].coef[ii][2] = GLV(line,6);
}
fclose(infu);
sprintf(wrd,"%scal/AVPinv.%s.dat",nsxdir,utcode);
printf("Read '%s'.\n",wrd);
infu = fopen_read(wrd);
while (fgetline(line,infu)) {
  nso = GLV(line,1);
  ii  = GLV(line,2);
  AVP2[nso].xoffinv[ii]    = GLV(line,3);
  AVP2[nso].coefinv[ii][0] = GLV(line,4);
  AVP2[nso].coefinv[ii][1] = GLV(line,5);
  AVP2[nso].coefinv[ii][2] = GLV(line,6);
}
fclose(infu);


printf("ORIGINAL SlitOffset=%f\n",SlitOffset);

/* .........Centroiding LIKE WHAT IS DONE in traceAVP routine ............... */
/* .....BUT HERE WE USE NEW AVP2[] values ........... */

/* .....We also apply DAR correction ... */

for (iexp=0; iexp<nexp; ++iexp) {
  printf("\nChecking exposure '%s'.\n",ffile[iexp]);

/* Read image and clean image. */
  nsx_clear_image( &IA );
  strcpy(IA.file,ffile[iexp]);
  strcpy(IA.root,"");
  ii1 = cindex_reverse(IA.file,"/");
  ii2 = cindex(IA.file,".fits");
  if (ii2-1 > ii1+1) { substrcpy_terminate(IA.file,ii1+1,ii2-1,IA.root,0); }
  nsx_read_image( &IA, 1 );
  for (nso=3; nso<=7; ++nso) {
    if (nso == 7) { ecol=IA.nc/2; } else { ecol=IA.nc; }
    mw = nsx_minwidth(ecol,nso);
    SPX[nso].numpro = 1 + mw;
  }
  SlitOffset=0.;
  nsx_wide_profile_mash( IA );    /* determine vertical shift of orders, set global SlitOffset */
  printf("SlitOffset=%f\n",SlitOffset);
  nsx_clean_image( &IA, SPX, NoClean, NoHotClean, nsxdir );

/* Echo. */
  mw = nsx_minwidth(nc,3);
  printf("NEW: arcsec0[%d]=%f  (slit width is %6.2f arcseconds)\n",iexp,arcsec0[iexp],mw*ARCSEC_PER_PIXEL);

/* Each order */
  asinc = ARCSEC_PER_PIXEL;
  asmax = cnint(( asinc * (double)(mw+1) ));
  for (nso=3; nso<=7; ++nso) {
    if (nso == 7) { ecol=IA.nc/2; } else { ecol=IA.nc; }
    nnt=0;
    for (col=50; col<ecol-50; col=col+5) {
      ii1=col-5;
      ii2=col+5;
      nn=0;
      for (as=0.; as<asmax; as=as+asinc) {
        sum=0.; num=0.;
        for (ii=ii1; ii<ii2; ++ii) {
          edge = nsx_find_real_image_row( 1, ii, nso );
          rb1 = edge + nsx_AVPinv( AVP2, nso, ii, as, IA );
          rb2 = edge + nsx_AVPinv( AVP2, nso, ii, as+asinc, IA );
          imgsum = nsx_fractional_pixel_rb( IA.nc, IA.clnimg, rb1, rb2, ii );
          sum = sum + imgsum;
          num = num + 1.;
        }
        if (num < 1.) { printf("***error: num<1 \n"); exit(1); }
        xx[nn] = as + (asinc/2.);
        yy[nn] = sum / num;
        ++nn;
      }
      cen_as   = nsx_centroid2( nn, xx, yy, arcsec0[iexp], 0.8, 4, 1 );
      if (cen_as < -99.) {
        printf("nso=%d  col=%5d  nn=%4d  cen_as=%f\n",nso,col,nn,cen_as);
        outfu = fopen_write("junk.dat");
        for (ii=0; ii<nn; ++ii) {
          fprintf(outfu,"%f %f\n",xx[ii],yy[ii]);
        }
        fclose(outfu);
        cpauseit();
      }
      yyt[nnt] = cen_as - arcsec0[iexp];
      xxt[nnt] = (double)col;
      wwt[nnt] = 1.0;
      ++nnt;
    }

/* Fit for this echelle order */
    order = 2;
    maxiter = nnt / 10;
    nsx_fitpoly_reject(nnt,xxt,yyt,wwt,order,maxiter,5.,&xofft,coeft,0);

/* Echo */
    sprintf(wrd,"NEWtrace%d_%s.dat",nso,IA.root);
    outfu = fopen_write(wrd);
    for (ii=0; ii<nnt; ++ii) {
      ff = cpolyval(order+1,coeft,(xxt[ii]-xofft));
      fprintf(outfu,"%8.2f %12.5e %12.5e\n",xxt[ii],(ff/ARCSEC_PER_PIXEL)*1000.,(yyt[ii]/ARCSEC_PER_PIXEL)*1000.);
    }
    fclose(outfu);

  }

  nsx_free_image( &IA );
}


/* Free */
for (iexp=0; iexp<mexp; ++iexp) {
for (nso=3; nso<=7; ++nso) {
  free(cas[iexp][nso]);
}}
return;
}




/* ----------------------------------------------------------------------
  Trace AVP recalibration..  -tab 18may2018
*/
void nsx_TraceAVP( IMGtype IA, AVPtype AVP[], SPXtype SPX[], char nsxdir[] )
{
/**/
int mw,col,nnt,nn,ecol,ii,year,month,day,jj,nso,ii1,ii2;
int order,maxiter;
/**/
double ff,arcsec0,edge;
double xx[300],yy[300],imgsum,sum,num;
double zt,as,asinc,asmax,rb1,rb2;
double cen_as,censum;
double xofft,coeft[9];
double theta,xv,wave,Delta_Arcsec;
/**/
const int mmt = 800;
double xxt[mmt],yyt[mmt],wwt[mmt];
/**/
char wrd[300];
/**/
FILE *outfu;
/**/
IMGtype IMG;
/**/

/* For this routine, there is no DAR (Differential Atmospheric Refraction)
   correction (at least in initial step), so set .el to zero. */
IMG.el      = 0.;
IMG.rotposn = 0.;
IMG.parang  = 0.;


/* Echo. */
printf(".......................TraceAVP.................\n");
fprintf(logfu,".......................TraceAVP.................\n");

/* Determine arcsec0 (reference point by constructing 3..6 profile. */
asinc = ARCSEC_PER_PIXEL;
asmax = cnint(( asinc * (double)SPX[3].numpro ));
censum= 0.;
for (nso=3; nso<=6; ++nso) {
  nn=0;
  for (as=0.; as<asmax; as=as+asinc) {
    sum=0.; num=0.;
    for (col=0; col<IA.nc; ++col) {
      edge = nsx_find_real_image_row( 1, col, nso );
      rb1 = edge + nsx_AVPinv( AVP, nso, col, as, IMG );
      rb2 = edge + nsx_AVPinv( AVP, nso, col, as+asinc, IMG );
      imgsum = nsx_fractional_pixel_rb( IA.nc, IA.clnimg, rb1, rb2, col );
      sum = sum + imgsum;
      num = num + 1.;
    }
    xx[nn] = as + (asinc/2.);
    yy[nn] = sum / num;
    ++nn;
  }
  cen_as = nsx_centroid2( nn, xx, yy, SPX[0].pro_cen, 0.8, 4, 1 );
  censum = censum + cen_as;
}
arcsec0 = censum / 4.;
mw = nsx_minwidth(IA.nc,3);
printf("NOTE: arcsec0=%f  (slit width is %6.2f arcseconds)\n",arcsec0,mw*ARCSEC_PER_PIXEL);
fprintf(logfu,"NOTE: arcsec0=%f  (slit width is %6.2f arcseconds)\n",arcsec0,mw*ARCSEC_PER_PIXEL);

/* Centroiding */
asinc = ARCSEC_PER_PIXEL;
asmax = cnint(( asinc * (double)SPX[3].numpro ));
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IA.nc/2; } else { ecol=IA.nc; }

  nnt=0;
  for (col=50; col<ecol-50; col=col+5) {
    ii1=col-5;
    ii2=col+5;
    nn=0;
    for (as=0.; as<asmax; as=as+asinc) {
      sum=0.; num=0.;
      for (ii=ii1; ii<ii2; ++ii) {
        edge = nsx_find_real_image_row( 1, ii, nso );
        rb1 = edge + nsx_AVPinv( AVP, nso, ii, as, IMG );
        rb2 = edge + nsx_AVPinv( AVP, nso, ii, as+asinc, IMG );
        imgsum = nsx_fractional_pixel_rb( IA.nc, IA.clnimg, rb1, rb2, ii );
        sum = sum + imgsum;
        num = num + 1.;
      }
      xx[nn] = as + (asinc/2.);
      yy[nn] = sum / num;
      ++nn;
    }
    cen_as   = nsx_centroid2( nn, xx, yy, arcsec0, 0.8, 4, 1 );

/* Apply DAR correction (if elevation is known). */
    if (IA.el > 5.) {
      zt = 90. - IA.el;
      theta = IA.rotposn - IA.parang;
      xv  = (double)col - WSC[nso].xoff;
      wave= cpolyval(WSC[nso].order+1,WSC[nso].coef,xv);
      Delta_Arcsec = nsx_difatmref(wave,zt) * cos(degtorad(theta));
/* Here we subtract since we want to 'undo' the atmospheric refraction (along slit). */
      cen_as = cen_as - Delta_Arcsec;
if (col == 999800) { printf("#@# zt=%f  nso=%d  col=%5d  DA=%f\n",zt,nso,col,Delta_Arcsec); cpauseit(); }
    }

/*
    rowoff   = nsx_AVPinv( AVP, nso, col, arcsec0 );
    aspp     = nsx_AVP(AVP,nso,col,rowoff+0.5) - nsx_AVP(AVP,nso,col,rowoff-0.5);
    yyt[nnt] = (cen_as - arcsec0) / aspp;
*/
    yyt[nnt] = cen_as - arcsec0;
    xxt[nnt] = (double)col;
    wwt[nnt] = 1.0;
    ++nnt;
  }

/* Fit for this echelle order */
  order = 2;
  maxiter = nnt / 10;
  nsx_fitpoly_reject(nnt,xxt,yyt,wwt,order,maxiter,5.,&xofft,coeft,0);

/* Echo */
  sprintf(wrd,"trace%d_%s.dat",nso,IA.root);
  outfu = fopen_write(wrd);
  for (ii=0; ii<nnt; ++ii) {
    ff = cpolyval(order+1,coeft,(xxt[ii]-xofft));
    fprintf(outfu,"%8.2f %12.5e %12.5e\n",xxt[ii],(ff/ARCSEC_PER_PIXEL)*1000.,(yyt[ii]/ARCSEC_PER_PIXEL)*1000.);
  }
  fclose(outfu);

/* Arcsec shift for all columns. */
  sprintf(wrd,"%s.change%d",IA.root,nso);
  printf("Writing '%s'.\n",wrd);
  fprintf(logfu,"Writing '%s'.\n",wrd);
  outfu = fopen_write(wrd);
  for (col=0; col<ecol; ++col) {
    ff = cpolyval(order+1,coeft,((double)col - xofft));
    fprintf(outfu,"%5d %12.5e 0.\n",col,ff);
  }
  fclose(outfu);

}

/* Check .utshut */
if (cindex(IA.utshut,"1000") == 0) {
  printf("===warning: utshut='%s'\n",IA.utshut);
  ii = cindex(IA.file,".fits");
  if (ii > 0) {
    jj = cindex_reverse(IA.file,"/");
    if (ii-1 > jj+1) {
      substrcpy_terminate(IA.file,jj+1,ii-1,wrd,0);
      printf("wrd='%s'\n",wrd);
      year = cvalread0(wrd,1,2);
      month= cvalread0(wrd,3,4);
      day  = cvalread0(wrd,5,6);
      sprintf(IA.utshut,"%4.4d-%2.2d-%2.2dT00:00:00.000",year+2000,month,day);
    }
  }
  if (cindex(IA.utshut,"1000") == 0) {
    printf("***error: utshut='%s'\n",IA.utshut);
    exit(1);
  }
}

/* Append to info file. */
sprintf(wrd,"%5d %9.5f %9.5f UT=%s root=%s file=%s",IA.nc,arcsec0,SlitOffset,IA.utshut,IA.root,IA.file);
misc_append_to_file( "change.info", wrd );
printf("%s\n",wrd);
fprintf(logfu,"%s\n",wrd);

return;
}


/* ----------------------------------------------------------------------
  Choose best AVP files for date of exposure.
*/
void nsx_choose_best_AVP( IMGtype IMG, char nsxdir[], char UseAVP[] ) 
{
/**/
int ii,loii,nn;
/**/
double jd,first_jd,losep,sep;
/**/
char line[100];
char wrd[200];
/**/
const int maxnn = 100;
double jds[maxnn];
/**/
FILE *infu;
/**/
/* Available files. */
strcpy(UseAVP,"2017-11-02");
misc_delete_file("ncba.temp");
sprintf(wrd,"ls -1d %scal/AVP.*.info > ncba.temp",nsxdir);
system(wrd);
infu = fopen_read("ncba.temp");
nn=0;
while (fgetline(line,infu)) {
  ii = cindex(line,"AVP.");
  substrcpy_terminate(line,ii+4,ii+13,wrd,0); strcat(wrd,"T00:00:00.000");
  jds[nn] = misc_zulu_to_julian(wrd);
  ++nn;
  if (nn > maxnn-3) { printf("***error: ncbA: too many AVP.*.info files in '%scal'.\n",nsxdir); exit(1); }
}
fclose(infu);
misc_delete_file("ncba.temp");
first_jd = misc_zulu_to_julian("2017-11-02T00:00:00.000");
jd = IMG.jd;
if (jd < first_jd) jd = first_jd;
losep=9.e+20; loii=-1;
for (ii=0; ii<nn; ++ii) {
  sep = ABS(( jd - jds[ii] ));
  if (sep < losep) { losep=sep; loii=ii; }
}
if (loii > -1) { misc_julian_to_zulu( jds[loii], UseAVP ); UseAVP[10]='\0'; }
printf("UT date of exposure is '%s'.\n",IMG.utshut);
printf("Best AVP file match '%s'.\n",UseAVP);
return;
}


/* ----------------------------------------------------------------------
 Check header quantities such as the parallactic angle.

 Compute Position Angle (paralactic angle)..
 taken from SKY() subroutine in astro_lib.f in makee package.

  HA : Hour Angle  (0h at meridian, negative in east)
  AZI: Azimuth  (0 degrees due north, 90 deg. due east, ...)
  ZT : Zenith angle (True)  (0 degrees at zenith)
  PA : Position Angle  (0 deg. red end toward due south, 90 deg due west,...)
(In the triangle zenith—object—celestial pole, the parallactic angle will be 
 the position angle of the zenith at the celestial object.  So the PA points
 the blue end at the zenith. On a N(up) and E(right) plot, PA=0 degrees would
 be pointing up (or north) and PA=90 would be pointing right (or east).)

 NOTE: NIRES is mounted on the Keck 2 telescope.
*/
void nsx_header_check( IMGtype IMG )
{
/**/
double Dec,HAdeg,xp,yp,zp,x,y,z,azi,air,zt,za,a,b,c,r,pa;
double dar,wv[20];
int ii;
/**/

/* Echo and set. */
printf(" _ _ _  _ _ _  _ _ _  Header check _ _ _  _ _ _  _ _ _ \n");
Dec   = IMG.dec;
HAdeg = IMG.ha;

if (Dec < -90.) { printf("No RA,Dec info..\n"); return; }

/* Calculate AZImuth and Zenith angle (True) */
zp = sin(degtorad(Dec));
xp = cos(degtorad(Dec)) * sin(degtorad(HAdeg));
yp = cos(degtorad(Dec)) * cos(degtorad(HAdeg));
x  = xp;
z  = ( zp * sin(degtorad(KECK2LAT)) ) + ( yp * cos(degtorad(KECK2LAT)) );
y  = ( yp * sin(degtorad(KECK2LAT)) ) - ( zp * cos(degtorad(KECK2LAT)) );
if (z != 0.) { zt = radtodeg( atan( sqrt((x*x)+(y*y)) / z ) ); } else { zt = 90.0; }
if (y != 0.) {
  azi = radtodeg( atan(x/y) );
} else {
  if (x > 0.) { azi = 270.; } else { azi = 90.; }
}
if (y > 0.) { azi = 180. + azi; }
if (azi < 0.) { azi = 360. + azi; }

/*
  Now calculate position angle... (0 deg.=red end toward south, 90deg.=west)
  Need to use x,y,z alt-azimuth euclidean coordinates. and the spherical
  triangle rule: cos(a)=cos(b)cos(c)+sin(b)sin(c)cos(ANGLE) where ANGLE is
  opposite "side" a.    ..a,b,c are in radians..
*/
a = degtorad(90. - KECK2LAT);
b = acos( (y*(-1.) * cos(degtorad(KECK2LAT))) + (z * sin(degtorad(KECK2LAT))) );  
c = acos(z);
if ( (sin(b) == 0.)||(sin(c) == 0.) ) {
  pa = 999.0;
} else {
  r = ( cos(a) - (cos(b) * cos(c)) ) / ( sin(b) * sin(c) );
/* adjust for possible numerical(machine) problems */
  if (ABS((r)) > 1.0) {
    pa = 999.0;
    if ((r >=  1.0)&&(r <  1.001)) pa=0.;
    if ((r <= -1.0)&&(r > -1.001)) pa=180.0;
  } else {
    pa = radtodeg(( acos(r) ));
  }
  if (HAdeg < 0.) pa = -1. * pa;
}

za = gal_apparent_zenith_angle( zt );
if (za < 0.) za=zt;
air = gal_airmass( za );
printf("  RA=%9.5f  HA=%9.5f  Dec=%9.5f \n",IMG.ra,IMG.ha,IMG.dec);
printf("  zt   = %9.5f  [zenith]=%9.5f  dif=%9.5f \n",zt,90.-IMG.el,zt-(90.-IMG.el));
printf("  za   = %9.5f  [zenith]=%9.5f  dif=%9.5f \n",za,90.-IMG.el,za-(90.-IMG.el));
printf("  azi  = %9.5f  .az     =%9.5f  dif=%9.5f \n",azi,IMG.az,azi-IMG.az);
printf("  pa   = %9.5f  .parang =%9.5f  dif=%9.5f \n",pa,IMG.parang,pa-IMG.parang);



/* 
   Now calculate differential refraction relative to 12000 angstroms in arcsecs.
   (pressure=760 mm Hg, T=15 deg Celsius, dry air) from CRC and other sources
*/
wv[0] = 5000.;
wv[1] = 8000.;
wv[2] = 9000.;
wv[3] = 10000.;
wv[4] = 11000.;
wv[5] = 12000.;
wv[6] = 13000.;
wv[7] = 14000.;
wv[8] = 15000.;
wv[9] = 16000.;
wv[10]= 17000.;
wv[11]= 18000.;
wv[12]= 19000.;
wv[13]= 20000.;
wv[14]= 21000.;
wv[15]= 22000.;
wv[16]= 23000.;
wv[17]= 24000.;
/*
x  = 12000.;
n5 = 2726.43 + ( 12.288 / (x*x*1.e-8))+ (0.3555 / (x*x*x*x*1.e-16));
n5 = n5 * 1.e-7;
for (ii=0; ii<18; ++ii) {
  x = wv[ii];
  n = 2726.43 + ( 12.288 / (x*x*1.e-8))+ (0.3555 / (x*x*x*x*1.e-16));
  n = n * 1.e-7;
  dar = 2.06265e+5 * (n - n5) * tan(degtorad(zt));
  printf("x = %12.2f  dar = %12.5f \n",x,dar);
}
*/

for (ii=0; ii<18; ++ii) {
  dar = nsx_difatmref( wv[ii], zt );
  printf(" %8.1f ang. rel. to. %8.1f ang (ref) : dar=%8.4f (%8.4f px)\n",wv[ii],DAR_refwave,dar,dar/ARCSEC_PER_PIXEL);
}

printf(" _ _ _  _ _ _  _ _ _  _ _ _ _ _ _  _ _ _  _ _ _  _ _ _ \n");



/* #@#
outfu = fopen_write("dar_check.dat");
el[0] = 57.6478037;  /x 0047 x/
el[1] = 57.4656532;  /x 0049 x/
el[2] = 56.9774166;  /x 0051 x/
el[3] = 56.7951375;  /x 0053 x/
for (ii=0; ii<18; ++ii) {
  fprintf(outfu,"\nwave=%f\n",wv[ii]);
  for (jj=0; jj<4; ++jj) {
    zt = 90. - el[jj];
    dar = nsx_difatmref( wv[ii], zt );
    fprintf(outfu," [zt=%8.4f] %8.1f ang. rel. to. %8.1f ang (ref) : dar=%8.4f (%8.4f px)\n",
       zt,wv[ii],DAR_refwave,dar,dar/ARCSEC_PER_PIXEL);
  }
}
fclose(outfu);
   #@# */


return;
}


/* ----------------------------------------------------------------------
  Test the profile centroid of bright star vs. RA and DEC to
  investigate arcseconds per pixel and position angle of slit.  -tab 24jul2018
*/
void nsx_TestProCent( IMGtype IA, AVPtype AVP[], SPXtype SPX[] )
{
/**/
int hiii,ii,jj,nso,nn;
int nsos[1000];
/**/
double cent,rowoff,xx[3000],yy[3000];
double ayp,azp,ra,dec,cx,cy,cz,xp,yp,zp,sum2,sum3,num,ra0,dec0;
double angle,dx,dy,rms,sep,rowsep;
double ras[1000],decs[1000],rows[1000];
/**/
RMtype RM;
/**/
char wrd[200];
char line[200];
/**/
FILE *infu;
FILE *outfu;
/**/
IMGtype IMG;
/**/

/* For this routine, there is no DAR (Differential Atmospheric Refraction)
   correction, so set .el to zero. */
IMG.el      = 0.;
IMG.rotposn = 0.;
IMG.parang  = 0.;


/* Check last test.. */
if (FileExist("tpc.prime")) {
  printf("Analyzing tpc.prime ...\n");
  infu = fopen_read("tpc.prime");
  sum2=0.; num=0.; nn=0;
  nn=0;
  while (fgetline(line,infu)) {
    ras[nn] = GLV(line,3);
    decs[nn]= GLV(line,4);
    rows[nn]= GLV(line,5);
    nsos[nn]= GLV(line,6);
    ++nn;
  }
  fclose(infu);
  sum2=0.; num=0.;
  outfu = fopen_write("tpc.prime.aspp.tbl");
  fprintf(outfu,"| sep     | rowsep  | aspp    |nso|\n");
  for (ii=0; ii<nn; ++ii) {
  for (jj=0; jj<nn; ++jj) {
    if (ii != jj) {
    if (nsos[ii] == nsos[jj]) {
      sep = cangsep(ras[ii],decs[ii],ras[jj],decs[jj]) * 3600.;
      if (sep > 3.0) {
        rowsep = ABS((rows[ii] - rows[jj]));
        fprintf(outfu," %9.6f %9.5f %9.6f %3d\n",sep,rowsep,sep/rowsep,nsos[ii]);
        yy[nn] = sep/rowsep;
        ++nn;
        sum2 = sum2 + (sep/rowsep);
        num  = num  + 1.;
      }
    }
    }
  }
  }
  fclose(outfu);
  sum3 = sum2 / num;
  rms = 0.;
  for (ii=0; ii<nn; ++ii) { rms = rms + ( (yy[ii] - sum3) * (yy[ii] - sum3) ); }
  rms = sqrt(( rms / (double)nn ));
  printf("Average ASPP = %9.5f   rms=%9.5f  num=%f\n",sum3,rms,num);

/* For s180304 data only..  */
  dx = ABS(( -2.910 - 2.910 ));
  dy = ABS(( -0.730 - 0.730 ));
  angle = radtodeg(( atan2(dy,dx) ));
  printf("s180304:  dx=%9.5f   dy=%9.5f  angle=%9.4f  90plus=%9.4f\n",dx,dy,angle,angle+90.);

/* For s171214 data only..  */
  dx = ABS(( -0.316 - 0.316 ));
  dy = ABS(( -5.992 - 5.992 ));
  angle = radtodeg(( atan2(dy,dx) ));
  printf("s171214:  dx=%9.5f   dy=%9.5f  angle=%9.4f  90plus=%9.4f\n",dx,dy,angle,angle+90.);

  return;
}




for (nso=3; nso<=6; ++nso) {
  nn=0;
  for (ii=0; ii<SPX[nso].pro_apn; ++ii) {
    rowoff = nsx_AVPinv( AVP, nso, 1000, SPX[0].pro_apx[ii], IMG );
    xx[nn] = rowoff;
    yy[nn] = SPX[nso].pro_apymed[ii];
    ++nn;
  }
/* High point and centroid. */
  hiii = 0;
  for (ii=0; ii<nn; ++ii) {
    if (yy[ii] > yy[hiii]) { hiii = ii; }
  }
  cent = nsx_centroid2( nn, xx, yy, xx[hiii], 5., 4, 1 );
  sprintf(wrd,"%12.5f %12.7f %12.7f %3d %s",cent,IA.ra,IA.dec,nso,IA.root);
  misc_append_to_file( "tpc.info", wrd );
}

/* Final.. */
/*
if (strcmp(IA.root,"s180304_0054") == 0) {
*/
if (strcmp(IA.root,"s171214_0018") == 0) {

  printf("Final one..\n");
  infu = fopen_read("tpc.info");
  sum2=0.; sum3=0.; num=0.;
  while (fgetline(line,infu)) { if (line[0] != '|') {
    ra = GLV(line,2); sum2 = sum2 + ra;
    dec= GLV(line,3); sum3 = sum3 + dec;
    num = num + 1.;
  }}
  fclose(infu);
  ra0 = sum2 / num;
  dec0= sum3 / num;
  printf("Mean ra,dec = %f %f\n",ra0,dec0);
  SetRotationMatrix( degtorad(ra0), degtorad(dec0), &RM );

  infu = fopen_read("tpc.info");
  outfu= fopen_write("tpc.prime");
  while (fgetline(line,infu)) { if (line[0] != '|') {
    cent= GLV(line,1);
    ra  = GLV(line,2);
    dec = GLV(line,3);
    nso = GLV(line,4);
    radec2xyz(ra,dec,&cx,&cy,&cz);
    xyz2prime(cx,cy,cz,RM.mat,&xp,&yp,&zp);
    ayp = radtodeg(yp) * 3600.;
    azp = radtodeg(zp) * 3600.;
    fprintf(outfu," %9.3f %9.3f %12.7f %12.7f %12.5f %3d \n",ayp,azp,ra,dec,cent,nso);
  }}
  fclose(infu);
  fclose(outfu);

}

/*
outfu = fopen_write("tpc.dat");
for (ii=0; ii<nn; ++ii) {
  fprintf(outfu,"%f %f\n",xx[ii],yy[ii]);
}
fclose(outfu);
*/

return;
}


/* ----------------------------------------------------------------------
  Low Pixel finder.   -tab 21aug2018
  Read in flat field images and look for holes.. find matching holes in
  other images.. Use -LowPix to run..
*/
void nsx_LowPix()
{
/**/
int bad,ff,kk,lowcount,iii,jjj,narr,ii1,ii2,jj1,jj2,ppp,pixno,ii,jj;
/**/
double sum,num,rms,sigs,median,arr[900];
/**/
const int nc = nc_Nominal;  /* 2048 */
const int nr = nr_Nominal;  /* 1024 */
const int ni = 5;
/**/
float *fltimg[ni];
float *sigimg[ni];
float *lowimg;
/**/
char wrd[200];
char root[100];
/**/
FILE *infu;
/**/

printf("LowPix..\n");

for (ii=0; ii<ni; ++ii) {
  fltimg[ii] = (float *)calloc(((nc*nr)+1000),sizeof(float));
  sigimg[ii] = (float *)calloc(((nc*nr)+1000),sizeof(float));
}
lowimg = (float *)calloc(((nc*nr)+1000),sizeof(float));

/* Read images. */
infu = fopen_read("in.list");
for (ii=0; ii<ni; ++ii) {
  fgetline(root,infu); sprintf(wrd,"%s.fits",root);
  printf("Read '%s'\n",wrd);
  nsx_read_general_image( wrd, fltimg[ii], nc, nr );
}
fclose(infu);

/* Find pixels below median by at least X sigs. */
for (ff=0; ff<ni; ++ff) {

for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) { pixno = ii + (jj*nc); sigimg[ff][pixno]=0.;  }}

for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) {
pixno = ii + (jj*nc);
sigimg[ff][pixno]=0.;

if (fltimg[ff][pixno] < 1000.) {
/* Median box. */
  ii1= ii - 3; if (ii1 <    0) ii1=0;
  ii2= ii + 3; if (ii2 > nc-1) ii2=nc-1;
  jj1= jj - 3; if (jj1 <    0) jj1=0;
  jj2= jj + 3; if (jj2 > nr-1) jj2=nr-1;
  narr=0;
  for (iii=ii1; iii<=ii2; ++iii) {
  for (jjj=jj1; jjj<=jj2; ++jjj) {
    ppp = iii + (jjj * nc);
    if (ppp != pixno) {
      arr[narr] = fltimg[ff][ppp];
      ++narr;
    }
  }}
  median = cfind_median8(narr,arr);
  if (median > 500.) {
/* Compute RMS. */
    rms=0.;
    for (kk=0; kk<narr; ++kk) { rms = rms + ((arr[kk]-median)*(arr[kk]-median)); }
    rms = sqrt(( rms )) / (double)narr;
/* Filter out high pixels. */
    bad=0;
    for (kk=0; kk<narr; ++kk) { 
      sigs = (arr[kk] - median) / rms;
      if (sigs > 30.) { arr[kk] = median; bad=1; }
    }
    if (bad) {
      rms=0.;
      for (kk=0; kk<narr; ++kk) { rms = rms + ((arr[kk]-median)*(arr[kk]-median)); }
      rms = sqrt(( rms )) / (double)narr;
    }
/* Low enough? */
    sigs = (median - fltimg[ff][pixno]) / rms;
    if (sigs > 100.) {
      sigimg[ff][pixno] = sigs; 
    }
/* Always mark very low pixels.. */
    if ((fltimg[ff][pixno] <  10.)&&(median >  500.)) { sigimg[ff][pixno] = 1000.; }
    if ((fltimg[ff][pixno] <  50.)&&(median > 2000.)) { sigimg[ff][pixno] = 1000.; }
    if ((fltimg[ff][pixno] <  90.)&&(median > 4000.)) { sigimg[ff][pixno] = 1000.; }
  }
}}}


sprintf(wrd,"sigimg%d.fits",ff);  printf("write '%s'\n",wrd);
nsx_write_general_image( wrd, sigimg[ff], nc, nr );
}


/* Check all exposures. */
lowcount=0;
for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) {
  pixno = ii + (jj*nc);
  lowimg[pixno]=0.;
  sum=0.; num=0.;
  for (ff=0; ff<ni; ++ff) {
    if (sigimg[ff][pixno] > 1.) {
      sum = sum + sigimg[ff][pixno]; 
      num = num + 1.;
    }
  }
  if (num > 2.9) { ++lowcount;  lowimg[pixno] = sum / num; }
}}
nsx_write_general_image( "lowimg.fits", lowimg, nc, nr );
printf("wrote lowimg.fits ... lowcount=%d\n",lowcount);



free(lowimg);
for (ii=0; ii<ni; ++ii) { free(fltimg[ii]); free(sigimg[ii]); }
return;
}




/* ----------------------------------------------------------------------
  Hot Pixel finder.   -tab 16aug2018
  Read in the one dark image I have, find high sigma pixels, compare to
  flag images from 3 other real exposures (see nsx_clean_image).  If the dark
  high sigma pixel appears in all 3 flag images, assume it is a real hot pixel.
*/
void nsx_HotPix()
{
/**/
const int nc = nc_Nominal;  /* 2048 */
const int nr = nr_Nominal;  /* 1024 */
const int ni = 3;
/**/
int ok,kk,iii,jjj,narr,ii1,ii2,jj1,jj2,ppp,pixno,ii,jj;
/**/
double lim1,lim2,difa,median,diff,arr[900];
/**/
float *darkimg;
float *hotimg;
float *flgimg[ni];
/**/
char wrd[200];
char root[100];
/**/
FILE *infu;
/**/

darkimg = (float *)calloc(((nc*nr)+1000),sizeof(float));
hotimg  = (float *)calloc(((nc*nr)+1000),sizeof(float));
for (ii=0; ii<ni; ++ii) {
  flgimg[ii]  = (float *)calloc(((nc*nr)+1000),sizeof(float));
}

printf("HotPix..\n");

/* Read images. */
infu = fopen_read("in.list");
fgetline(root,infu); sprintf(wrd,"%s.fits",root);
printf("Read dark image '%s'\n",wrd);
nsx_read_general_image( wrd, darkimg, nc, nr );
for (ii=0; ii<ni; ++ii) {
  fgetline(root,infu); sprintf(wrd,"%s-flg.fits",root);
  printf("Read '%s'\n",wrd);
  nsx_read_general_image( wrd, flgimg[ii], nc, nr );
}
fclose(infu);

/* Clear. */
for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) {
  pixno = ii + (jj*nc);
  hotimg[pixno]=0.;
}}

/* Find pixels above median and above 10 sigs. */
for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) {
pixno = ii + (jj*nc);
if (hotimg[pixno] < 0.1) {

/* Median box. */
  ii1= ii - 3; if (ii1 <    0) ii1=0;
  ii2= ii + 3; if (ii2 > nc-1) ii2=nc-1;
  jj1= jj - 3; if (jj1 <    0) jj1=0;
  jj2= jj + 3; if (jj2 > nr-1) jj2=nr-1;
  narr=0;
  for (iii=ii1; iii<=ii2; ++iii) {
  for (jjj=jj1; jjj<=jj2; ++jjj) {
  if ((iii != ii)&&(jjj != jj)) {
    ppp = iii + (jjj * nc);
    arr[narr] = darkimg[ppp];
    ++narr;
  }}}
  median = cfind_median8(narr,arr);
  
/* Limits. */
  lim1 = 100.;
  if (median > 1000.) { lim1 = median / 10.; }
  lim2 = lim1 / 4.;

/* Compute RMS.
  rms=0.;
  for (kk=0; kk<narr; ++kk) { rms = rms + ((arr[kk]-median)*(arr[kk]-median)); }
  rms = sqrt(( rms )) / (double)narr;
*/

/* Hot enough? */
  diff = darkimg[pixno] - median;
  if (diff > lim1) { 
    hotimg[pixno] = diff; 

/* Check adjacent pixels. */
    if (diff > (10.*lim1)) {
      ii1= ii - 1; if (ii1 <    0) ii1=0;
      ii2= ii + 1; if (ii2 > nc-1) ii2=nc-1;
      jj1= jj - 1; if (jj1 <    0) jj1=0;
      jj2= jj + 1; if (jj2 > nr-1) jj2=nr-1;
      for (iii=ii1; iii<=ii2; ++iii) {
      for (jjj=jj1; jjj<=jj2; ++jjj) {
        ppp = iii + (jjj * nc);
        if (hotimg[ppp] < 0.1) {
          difa = darkimg[ppp] - median;
          if (difa > lim2) { hotimg[ppp] = difa;  }
        }
      }}
    }

  }
}}}

nsx_write_general_image( "hotimg.fits", hotimg, nc, nr );


/* Restrict hot pixels based on other exposures. */
for (ii=0; ii<nc; ++ii) {
for (jj=0; jj<nr; ++jj) {
pixno = ii + (jj*nc);
if (hotimg[pixno] > 0.1) {
  ok=0;
  for (kk=0; kk<ni; ++kk) {
    if (flgimg[kk][pixno] > 0.1) ++ok;
  }
  if (ok < 2) hotimg[pixno]=0.;
}}}

nsx_write_general_image( "hotimg2.fits", hotimg, nc, nr );

nsx_write_general_image( "HotPix.fits", hotimg, nc, nr );

free(darkimg);
free(hotimg);
for (ii=0; ii<ni; ++ii) { free(flgimg[ii]); }
return;
}


/* ----------------------------------------------------------------------
 De-Fractionalize a pixel.  Add a value into the pixels on an image in
 proportion to how much they cover.  cb1,cb2 (real) are column boundaries.
 and jj is the (integer) row number.  valu[] is the image with the value
 distributed among pixel(s), and vwgt[] is the weight image.
 Must have cb2 > cb1 ..
*/
void nsx_defract_pixel_cb( int nc, float valu[], float vwgt[], double cb1, double cb2, int jj, double value )
{
/**/
int iii,ii1,ii2,pixno;
double wgt;
/**/
ii1 = cnint(cb1);
ii2 = cnint(cb2);
if (ii1 == ii2) {
  wgt = cb2 - cb1;
  pixno = ii1 + (jj * nc);
  valu[pixno] = valu[pixno] + (value * wgt);
  vwgt[pixno] = vwgt[pixno] + (        wgt);
} else {
  wgt  = ((double)ii1 + 0.5) - cb1;
  pixno= ii1 + (jj * nc);
  valu[pixno] = valu[pixno] + (value * wgt);
  vwgt[pixno] = vwgt[pixno] + (        wgt);
  wgt  = cb2 - ((double)ii2 - 0.5);
  pixno= ii2 + (jj * nc);
  valu[pixno] = valu[pixno] + (value * wgt);
  vwgt[pixno] = vwgt[pixno] + (        wgt);
  for (iii=ii1+1; iii<ii2; ++iii) {
    pixno= iii + (jj * nc);
    valu[pixno] = valu[pixno] + (value * 1.0);
    vwgt[pixno] = vwgt[pixno] + (        1.0);
  }
}
return;
}


/* ----------------------------------------------------------------------
 De-Fractionalize a pixel.  Add a value into the pixels on an image in
 proportion to how much they cover.  rb1,rb2 (real) are row boundaries.
 and ii is the (integer) column number.  valu[] is the image with the value
 distributed among pixel(s), and vwgt[] is the weight image.
 Must have rb2 > rb1 ..
*/
void nsx_defract_pixel_rb( int nc, float valu[], float vwgt[], double rb1, double rb2, int ii, double value )
{
/**/
int jjj,jj1,jj2,pixno;
double wgt;
/**/
jj1 = cnint(rb1);
jj2 = cnint(rb2);
if (jj1 == jj2) {
  wgt = rb2 - rb1;
  pixno = ii + (jj1 * nc);
  valu[pixno] = valu[pixno] + (value * wgt);
  vwgt[pixno] = vwgt[pixno] + (        wgt);
} else {
  wgt  = ((double)jj1 + 0.5) - rb1;
  pixno= ii + (jj1 * nc);
  valu[pixno] = valu[pixno] + (value * wgt);
  vwgt[pixno] = vwgt[pixno] + (        wgt);
  wgt  = rb2 - ((double)jj2 - 0.5);
  pixno= ii + (jj2 * nc);
  valu[pixno] = valu[pixno] + (value * wgt);
  vwgt[pixno] = vwgt[pixno] + (        wgt);
  for (jjj=jj1+1; jjj<jj2; ++jjj) {
    pixno= ii + (jjj * nc);
    valu[pixno] = valu[pixno] + (value * 1.0);
    vwgt[pixno] = vwgt[pixno] + (        1.0);
  }
}
return;
}


/* ----------------------------------------------------------------------
  Coadd and check flat template..   -tab 30aug2018
  Reads 'flat.ls' file.
  ( First run 'nsx s180304_0020.fits -buildflat' on all appropriate flat
    fields (consecutive exposures same exposure times) and then run
    'nsx s180304_0020.fits -checkflat' (file is just to get SPX[] ..) )
*/
void nsx_checkflat_template( IMGtype IA, AVPtype AVP[], SPXtype SPX[], 
                             SOPtype SOP1[], SOPtype SOP2[], char nsxdir[] )
{
/**/
char wrd[200];
char root[200];
/**/
float *caf;
float *cnf;
float *img;
/**/
int ii,count,nc,nr;
int ecol,narr,nso,kk,col,jj,ii1,ii2,pp;
/**/
double s2n,s2nf,rms,median,medianf,edge,arf[9000],arr[9000];
/**/
FILE *infu;
FILE *outfu;
/**/

/* Allocate. */
nc=IA.nc; nr=IA.nr;
caf = (float *)calloc(((nc*nr)+1000),sizeof(float));
cnf = (float *)calloc(((nc*nr)+1000),sizeof(float));
img = (float *)calloc(((nc*nr)+1000),sizeof(float));

/* Add up cleaned flats. */
count=0;
for (ii=0; ii<(nc*nr); ++ii) { caf[ii]=0.; }
infu = fopen_read("flat.ls");
while (fgetline(root,infu)) {
  ii=cindex(root,".fits"); root[ii]='\0';
  sprintf(wrd,"%s-clnimg.fits",root); printf("Read %s \n",wrd);
  nsx_read_general_image( wrd, img, nc, nr );
  for (ii=0; ii<(nc*nr); ++ii) { caf[ii] = caf[ii] + img[ii]; }
  ++count;
}
fclose(infu);
for (ii=0; ii<(nc*nr); ++ii) { caf[ii] = caf[ii] / (float)count; }
printf("write flatave.fits  count=%d\n",count);
nsx_write_general_image( "flatave.fits", caf, nc, nr );
count=0;
for (ii=0; ii<(nc*nr); ++ii) { if (caf[ii] < 1.0) { ++count; caf[ii]=1.0; } }
printf("found %d low pixel(s) in caf\n",count);

/* Add up normalized flats. */
count=0;
for (ii=0; ii<(nc*nr); ++ii) { cnf[ii]=0.; }
infu = fopen_read("flat.ls");
while (fgetline(root,infu)) {
  ii=cindex(root,".fits"); root[ii]='\0';
  sprintf(wrd,"%s-comb.fits",root); printf("Read %s \n",wrd);
  nsx_read_general_image( wrd, img, nc, nr );
  for (ii=0; ii<(nc*nr); ++ii) { cnf[ii] = cnf[ii] + img[ii]; }
  ++count;
}
fclose(infu);
for (ii=0; ii<(nc*nr); ++ii) { cnf[ii] = cnf[ii] / (float)count; }
printf("write flatnorm.fits\n");
nsx_write_general_image( "flatnorm.fits", cnf, nc, nr );
count=0;
for (ii=0; ii<(nc*nr); ++ii) { if (cnf[ii] < 0.001) { ++count; } }
printf("found %d low pixel(s) in cnf\n",count);
  

/* Divide each norm flat by total norm and look at noise. */
infu = fopen_read("flat.ls");
while (fgetline(root,infu)) {

  ii=cindex(root,".fits"); root[ii]='\0';
  sprintf(wrd,"%s-comb.fits",root); printf("Read %s \n",wrd);
  nsx_read_general_image( wrd, img, nc, nr );
  for (ii=0; ii<(nc*nr); ++ii) { img[ii] = img[ii] / cnf[ii]; }
  sprintf(wrd,"%s-div.fits",root); printf("write %s \n",wrd);
  nsx_write_general_image( wrd, img, nc, nr );


/* Just use first flat for this check. */

/* THIS checks the S/N from the flat (assuming eperdn=1.0) and compares to
 * the S/N in first normalized flat divided by the total normalized flat.
 * The S/N is larger by about 2.0-- this is probably due to smoothing.. */

/* NOTE: I have not included the correct S/N accounting for error in total flat norm .. */

  if (strcmp(root,"s180304_0020")==0) {
    for (nso=3; nso<=7; ++nso) {
      sprintf(wrd,"jmdn%d.dat",nso);
      outfu = fopen_write(wrd);
      if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
      for (col=10; col<ecol-10; col=col+10) {
        ii1 = col - 5;
        ii2 = col + 5;
        narr=0;
        for (ii=ii1; ii<ii2; ++ii) {
          edge= nsx_find_real_image_row( 1, ii, nso );
          for (kk=5; kk<SPX[nso].numpro-5; ++kk) {
            jj = cnint(edge) + kk;
            pp = ii + (jj * nc);
            arr[narr] = img[pp];
            arf[narr] = caf[pp];
            ++narr;
          }
        }
        median = cfind_median8(narr,arr);
        medianf= cfind_median8(narr,arf);
        rms=0.;
        for (ii=0; ii<narr; ++ii) {
          rms = rms + ( (arr[ii] - median) * (arr[ii] - median) );
        }
        rms = sqrt(( rms / (double)narr ));
        s2n = median / rms;
        s2nf= (medianf) / sqrt((medianf));
        fprintf(outfu," %5d %12.2f %12.7f %12.7f %12.7f %12.7f %12.7f \n",col,medianf,1000.*(median - 1.0),rms,s2n,s2nf,s2n/s2nf);
      }
      fclose(outfu);
    }
  }

}
fclose(infu);



free(caf);
free(cnf);
free(img);
return;
}



/* ----------------------------------------------------------------------
  Build flat template..   -tab 20aug2018
  Process flat in stages:

Type A smoothing-- good for perserving small features.. best for flat areas..
   1--- correct hot and low pixels and any CRs  (clean_image)
   2--- first smooth the flat field uses square box of 9 x 9 (col/rowrad0)..
        Create 'Adivi1' = (clnimg / smooth).
   3--- smooth Adivi1 by finding an average value in each slanted column position,
        and expanding that value along each slant.  To create the proper smoothed 
        image the pixel value must be 'defractionalized' along each slant. 
        This takes almost all the rest of the abs.line effects.
        Create 'Bdivi2' = Bdivi1 / (valu/vwgt)
   4--- Find medians in 100 column segments aligned along the slit position 
        (following the 'object' features along the curved slit traces) and
        divide into 'Adivi2'.  This takes out the 'object' features (make Adivi3).

Type B smoothing-- good for taking out absorption lines.
   1--- correct hot and low pixels and any CRs  (clean_image)
   2--- first smooth the flat field uses boxes of 1 x 5 (rowrad1) which are
        aligned along the slant. This takes out most of the abs.line features.
        Create 'Bdivi1' = (clnimg / smooth).
   3--- smooth Bdivi1 by finding an average value in each slanted column position,
        and expanding that value along each slant.  To create the proper smoothed 
        image the pixel value must be 'defractionalized' along each slant. 
        This takes almost all the rest of the abs.line effects.
        Create 'Bdivi2' = Bdivi1 / (valu/vwgt)
   4--- do another small box 1 x 9(rowrad2) smoothing along each slant (like step 2).
        This takes out the rest of the abs.line effects.
        Create 'Bdivi3' = Bdivi2 / smooth .
   5--- Find medians in 100 column segments aligned along the slit position 
        (following the 'object' features along the curved slit traces) and
        divide into 'Bdivi3'.  This takes out the 'object' features (make Bdivi4).

        
*/
void nsx_buildflat_template( IMGtype IA, AVPtype AVP[], SPXtype SPX[], 
                             SOPtype SOP1[], SOPtype SOP2[], char nsxdir[] )
{
/**/
char wrd[200];
char line[200];
/**/
int kk,narr,ii,pp,pixno,nc,nr,jj,ecol,nso,col,row;
int ccol,col1,col2,rowtop,rowbot,row1,row2;
int smty,ff,nfa,fa_nso[30],fa_sc[30],fa_ec[30],fa_smty[30];
/**/
int colrad0 = 4;
int rowrad0 = 4;
int rowrad1 = 8;
int rowrad2 =11;
/**/
double sum,num,rcol,edge,rowoff,cb1,cb2,imgsum,offset,offset0;
double median,arr[3000],rb1,rb2,prof[300];
/**/
float *Adivi1;
float *Adivi2;
float *Adivi3;
float *Bdivi1;
float *Bdivi2;
float *Bdivi3;
float *Bdivi4;
float *valu;
float *vwgt;
/**/
FILE *infu;
/**/

/* Set. */
nc = IA.nc;
nr = IA.nr;
Adivi1=(float *)calloc(((nc*nr)+1000),sizeof(float));
Adivi2=(float *)calloc(((nc*nr)+1000),sizeof(float));
Adivi3=(float *)calloc(((nc*nr)+1000),sizeof(float));
Bdivi1= (float *)calloc(((nc*nr)+1000),sizeof(float));
Bdivi2= (float *)calloc(((nc*nr)+1000),sizeof(float));
Bdivi3= (float *)calloc(((nc*nr)+1000),sizeof(float));
Bdivi4= (float *)calloc(((nc*nr)+1000),sizeof(float));
valu = (float *)calloc(((nc*nr)+1000),sizeof(float));
vwgt = (float *)calloc(((nc*nr)+1000),sizeof(float));


/* ....... Type 1 smoothing: use this away from strong abs.lines. ........... */
for (ii=0; ii<(nc*nr); ++ii) { Adivi1[ii]=0.; Adivi2[ii]=0.; Adivi3[ii]=0.; }

/* Smooth 'clnimg' in boxes bounded by slit edges. */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (col=0; col<ecol; col=col+1) {
    col1 = col - colrad0;
    col2 = col + colrad0;
    if (col1 < 0   ) col1=0;
    if (col2 > nc-1) col2=nc-1;
    edge = nsx_find_real_image_row( 1, col, nso );
    rowbot = cnint(edge) + 1;
    rowtop = rowbot + SPX[nso].numpro - 1;
    for (row=rowbot; row<=rowtop; row=row+1) {
      row1 = row - rowrad0;
      row2 = row + rowrad0;
      if (row1 < rowbot) row1=rowbot;
      if (row2 > rowtop) row2=rowtop;
      narr=0;
      for (ii=col1; ii<=col2; ++ii) {
      for (jj=row1; jj<=row2; ++jj) {
        pp = ii + (jj * nc);
        arr[narr] = IA.clnimg[pp];
        ++narr;
      }}
      median = cfind_median8(narr,arr);
      pixno = col + (row * nc);
      if (median > 0.) { Adivi1[pixno] = IA.clnimg[pixno] / median; }
    }
  }
}

/* Smooth Adivi1[] using whole slant slices. 'Defract' median along slant slice. */
for (ii=0; ii<(nc*nr); ++ii) { valu[ii]=0.; vwgt[ii]=0.; }
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (col=0; col<ecol; col=col+1) {
    edge = nsx_find_real_image_row( 1, col, nso );
    rowbot = cnint(edge) + 1;
    rowtop = rowbot + SPX[nso].numpro - 1;
    rowoff = (double)rowbot - edge;
    offset0= nsx_SOP2( SOP1, SOP2, nso, col, rowoff );
    narr=0;
    for (jj=rowbot; jj<=rowtop; ++jj) {
      rowoff = (double)jj - edge;
      offset = nsx_SOP2( SOP1, SOP2, nso, col, rowoff );
      rcol= (double)col + (offset - offset0);
      cb1 = rcol - 0.5;
      cb2 = rcol + 0.5;
      nsx_fractional_pixel_cb( nc, Adivi1, cb1, cb2, jj, &imgsum );
      sum = sum + imgsum;
      num = num + 1.;
      arr[narr] = imgsum;
      ++narr;
    }
    median= cfind_median8(narr,arr);
    for (jj=rowbot; jj<=rowtop; ++jj) {
      rowoff = (double)jj - edge;
      offset = nsx_SOP2( SOP1, SOP2, nso, col, rowoff );
      rcol= (double)col + (offset - offset0);
      cb1 = rcol - 0.5;
      cb2 = rcol + 0.5;
      nsx_defract_pixel_cb( nc, valu, vwgt, cb1, cb2, jj, median );
    }
  }
}
/* Divide Adivi1[] by normalized valu[] image. */
for (ii=0; ii<(nc*nr); ++ii) { 
  if (vwgt[ii] > 0.) { 
    valu[ii] = valu[ii] / vwgt[ii]; 
    Adivi2[ii]= Adivi1[ii] / valu[ii];
  }
}

/* ...Profile correction ... */
for (ii=0; ii<(nc*nr); ++ii) { valu[ii]=0.; vwgt[ii]=0.; }
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (ccol=0; ccol<ecol; ccol=ccol+50) {
    col1 = ccol - 50;
    col2 = ccol + 50;
    if (col1 < 0     ) col1=0;
    if (col2 > ecol-1) col2=ecol-1;
    for (kk=0; kk<SPX[nso].numpro; ++kk) {
      narr=0;
      for (col=col1; col<col2; ++col) {
        edge= nsx_find_real_image_row( 1, col, nso );
        rb1 = edge + (double)kk - 0.5;
        rb2 = rb1 + 1.0;
        arr[narr] = nsx_fractional_pixel_rb( nc, Adivi2, rb1, rb2, col );
        ++narr;
      }
      prof[kk] = cfind_median8(narr,arr);
    }
    for (kk=0; kk<SPX[nso].numpro; ++kk) {
    if (prof[kk] > 0.) {
      for (col=col1; col<col2; ++col) {
        edge= nsx_find_real_image_row( 1, col, nso );
        rb1 = edge+ (double)kk - 0.5;
        rb2 = rb1 + 1.0;
        nsx_defract_pixel_rb( nc, valu, vwgt, rb1, rb2, col, prof[kk] );
      }
    }
    }
  }
}
/* Divide Adivi2[] by normalized valu[] image. */
for (ii=0; ii<(nc*nr); ++ii) { 
  if ((vwgt[ii] > 0.)&&(valu[ii] > 0.)) { 
    valu[ii] = valu[ii]  / vwgt[ii]; 
    Adivi3[ii]= Adivi2[ii] / valu[ii];
  }
}

sprintf(wrd,"%s-Adivi1.fits" ,IA.root);  nsx_write_general_image( wrd, Adivi1,nc, nr );      printf("Writing '%s'.\n",wrd);
sprintf(wrd,"%s-Adivi2.fits" ,IA.root);  nsx_write_general_image( wrd, Adivi2,nc, nr );      printf("Writing '%s'.\n",wrd);
sprintf(wrd,"%s-Adivi3.fits" ,IA.root);  nsx_write_general_image( wrd, Adivi3,nc, nr );      printf("Writing '%s'.\n",wrd);



/* ..Type 2 smoothing: use this near strong abs.lines. ........... */
for (ii=0; ii<(nc*nr); ++ii) { Bdivi1[ii]=0.; Bdivi2[ii]=0.; Bdivi3[ii]=0.; Bdivi4[ii]=0.; }

/* Smooth 'clnimg' along slant for each pixel. Create Bdivi1[] = clnimg / smooth .. */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (col=0; col<ecol; col=col+1) {
    edge = nsx_find_real_image_row( 1, col, nso );
    rowbot = cnint(edge) + 1;
    rowtop = rowbot + SPX[nso].numpro - 1;
    for (row=rowbot; row<=rowtop; row=row+1) {
      rowoff = (double)row - edge;
      offset0= nsx_SOP2( SOP1, SOP2, nso, col, rowoff );
      row1 = row - rowrad1;
      row2 = row + rowrad1;
      if (row1 < rowbot) row1=rowbot;
      if (row2 > rowtop) row2=rowtop;
      narr=0;
      for (jj=row1; jj<=row2; ++jj) {
        rowoff = (double)jj - edge;
        offset = nsx_SOP2( SOP1, SOP2, nso, col, rowoff );
        rcol= (double)col + (offset - offset0);
        cb1 = rcol - 0.5;
        cb2 = rcol + 0.5;
        nsx_fractional_pixel_cb( nc, IA.clnimg, cb1, cb2, jj, &imgsum );
        arr[narr] = imgsum;
        ++narr;
      }
      median = cfind_median8(narr,arr);
      pixno = col + (row * nc);
      Bdivi1[pixno] = IA.clnimg[pixno] / median;
    }
  }
}

/* Smooth Bdivi1[] using whole slant slices. 'Defract' median along slant slice. */
for (ii=0; ii<(nc*nr); ++ii) { valu[ii]=0.; vwgt[ii]=0.; }
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (col=0; col<ecol; col=col+1) {
    edge = nsx_find_real_image_row( 1, col, nso );
    rowbot = cnint(edge) + 1;
    rowtop = rowbot + SPX[nso].numpro - 1;
    rowoff = (double)rowbot - edge;
    offset0= nsx_SOP2( SOP1, SOP2, nso, col, rowoff );
    narr=0;
    for (jj=rowbot; jj<=rowtop; ++jj) {
      rowoff = (double)jj - edge;
      offset = nsx_SOP2( SOP1, SOP2, nso, col, rowoff );
      rcol= (double)col + (offset - offset0);
      cb1 = rcol - 0.5;
      cb2 = rcol + 0.5;
      nsx_fractional_pixel_cb( nc, Bdivi1, cb1, cb2, jj, &imgsum );
      arr[narr] = imgsum;
      ++narr;
    }
    median= cfind_median8(narr,arr);
    for (jj=rowbot; jj<=rowtop; ++jj) {
      rowoff = (double)jj - edge;
      offset = nsx_SOP2( SOP1, SOP2, nso, col, rowoff );
      rcol= (double)col + (offset - offset0);
      cb1 = rcol - 0.5;
      cb2 = rcol + 0.5;
      nsx_defract_pixel_cb( nc, valu, vwgt, cb1, cb2, jj, median );
    }
  }
}
/* Divide Bdivi1[] by normalized valu[] image. */
for (ii=0; ii<(nc*nr); ++ii) { 
  if (vwgt[ii] > 0.) { 
    valu[ii] = valu[ii] / vwgt[ii]; 
    Bdivi2[ii]= Bdivi1[ii] / valu[ii];
  }
}

/* AGAIN: Smooth 'Bdivi2' along slant for each pixel. Create Bdivi3[] = Bdivi2 / smooth .. */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (col=0; col<ecol; col=col+1) {
    edge = nsx_find_real_image_row( 1, col, nso );
    rowbot = cnint(edge) + 1;
    rowtop = rowbot + SPX[nso].numpro - 1;
    for (row=rowbot; row<=rowtop; row=row+1) {
      rowoff = (double)row - edge;
      offset0= nsx_SOP2( SOP1, SOP2, nso, col, rowoff );
      row1 = row - rowrad2;
      row2 = row + rowrad2;
      if (row1 < rowbot) row1=rowbot;
      if (row2 > rowtop) row2=rowtop;
      narr=0;
      for (jj=row1; jj<=row2; ++jj) {
        rowoff = (double)jj - edge;
        offset = nsx_SOP2( SOP1, SOP2, nso, col, rowoff );
        rcol= (double)col + (offset - offset0);
        ii  = cnint(rcol);
        pp  = ii + (jj * nc);
        if (Bdivi2[pp] > 0.) {
          cb1 = rcol - 0.5;
          cb2 = rcol + 0.5;
          nsx_fractional_pixel_cb( nc, Bdivi2, cb1, cb2, jj, &imgsum );
          arr[narr] = imgsum;
          ++narr;
        }
      }
      median = cfind_median8(narr,arr);
      pixno = col + (row * nc);
      Bdivi3[pixno] = Bdivi2[pixno] / median;
    }
  }
}

/* ...Profile correction ... */
for (ii=0; ii<(nc*nr); ++ii) { valu[ii]=0.; vwgt[ii]=0.; }
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (ccol=0; ccol<ecol; ccol=ccol+50) {
    col1 = ccol - 50;
    col2 = ccol + 50;
    if (col1 < 0     ) col1=0;
    if (col2 > ecol-1) col2=ecol-1;
    for (kk=0; kk<SPX[nso].numpro; ++kk) {
      narr=0;
      for (col=col1; col<col2; ++col) {
        edge= nsx_find_real_image_row( 1, col, nso );
        rb1 = edge + (double)kk - 0.5;
        rb2 = rb1 + 1.0;
        arr[narr] = nsx_fractional_pixel_rb( nc, Bdivi3, rb1, rb2, col );
        ++narr;
      }
      prof[kk] = cfind_median8(narr,arr);
    }
    for (kk=0; kk<SPX[nso].numpro; ++kk) {
    if (prof[kk] > 0.) {
      for (col=col1; col<col2; ++col) {
        edge= nsx_find_real_image_row( 1, col, nso );
        rb1 = edge+ (double)kk - 0.5;
        rb2 = rb1 + 1.0;
        nsx_defract_pixel_rb( nc, valu, vwgt, rb1, rb2, col, prof[kk] );
      }
    }
    }
  }
}
/* Divide Bdivi3[] by normalized valu[] image. */
for (ii=0; ii<(nc*nr); ++ii) { 
  if ((vwgt[ii] > 0.)&&(valu[ii] > 0.)) { 
    valu[ii] = valu[ii]  / vwgt[ii]; 
    Bdivi4[ii]= Bdivi3[ii] / valu[ii];
  }
}

sprintf(wrd,"%s-Bdivi1.fits" ,IA.root);  nsx_write_general_image( wrd, Bdivi1,nc, nr );      printf("Writing '%s'.\n",wrd);
sprintf(wrd,"%s-Bdivi2.fits" ,IA.root);  nsx_write_general_image( wrd, Bdivi2,nc, nr );      printf("Writing '%s'.\n",wrd);
sprintf(wrd,"%s-Bdivi3.fits" ,IA.root);  nsx_write_general_image( wrd, Bdivi3,nc, nr );      printf("Writing '%s'.\n",wrd);
sprintf(wrd,"%s-Bdivi4.fits" ,IA.root);  nsx_write_general_image( wrd, Bdivi4,nc, nr );      printf("Writing '%s'.\n",wrd);
sprintf(wrd,"%s-valu.fits"  ,IA.root);  nsx_write_general_image( wrd, valu, nc, nr );      printf("Writing '%s'.\n",wrd);
sprintf(wrd,"%s-vwgt.fits"  ,IA.root);  nsx_write_general_image( wrd, vwgt, nc, nr );      printf("Writing '%s'.\n",wrd);
sprintf(wrd,"%s-clnimg.fits",IA.root);  nsx_write_general_image( wrd, IA.clnimg, nc, nr ); printf("Writing '%s'.\n",wrd);

for (ii=0; ii<(nc*nr); ++ii) { 
  valu[ii]=0.;
  if (Bdivi4[ii] > 0.) { valu[ii] = Adivi3[ii] / Bdivi4[ii]; }
}
sprintf(wrd,"%s-Achck.fits" ,IA.root);  nsx_write_general_image( wrd, valu, nc, nr );      printf("Writing '%s'.\n",wrd);


/* Combine Adivi3 and Bdivi4 and trim.. */
for (ii=0; ii<(nc*nr); ++ii) { 
  valu[ii]=1.0; 
  if (Adivi3[ii] < 0.001) Adivi3[ii]=1.0;
  if (Bdivi4[ii] < 0.001) Bdivi4[ii]=1.0;
}
sprintf(wrd,"%s/cal/flatabs.dat",nsxdir);
infu = fopen_read(wrd);
nfa=0;
while (fgetline(line,infu)) {
  fa_nso[nfa] = GLV(line,1);
  fa_sc[nfa]  = GLV(line,2);
  fa_ec[nfa]  = GLV(line,3);
  fa_smty[nfa]= GLV(line,4);
  ++nfa;
}
fclose(infu);
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (ii=5; ii<ecol-6; ++ii) {
    smty = 1;
    for (ff=0; ff<nfa; ++ff) {
      if (nso == fa_nso[ff]) {
        if ((ii >= fa_sc[ff])&&(ii <= fa_ec[ff])) { smty = fa_smty[ff]; }
      }
    }
    edge= nsx_find_real_image_row( 1, ii, nso );
    for (kk=3; kk<SPX[nso].numpro-1; ++kk) {
      jj = cnint(edge) + kk;
      pp = ii + (jj * nc);
      if (smty == 1) { valu[pp] = Adivi3[pp]; } else { valu[pp] = Bdivi4[pp]; } 
    }
  }
}


/* Clean up lower edges of blue orders. */
nso = 6;
for (ii=nc/2; ii<nc; ++ii) {
  edge = nsx_find_real_image_row( 1, ii, nso );
  jj = cnint(edge) + 3; pp = ii + (jj * nc); valu[pp]=1.0;
}
nso = 7;
for (ii=0; ii<nc/2; ++ii) {
  edge = nsx_find_real_image_row( 1, ii, nso );
  jj = cnint(edge) + 3; pp = ii + (jj * nc); valu[pp]=1.0;
}

/* No NANs. */
for (ii=0; ii<(nc*nr); ++ii) { if (valu[ii] != valu[ii]) valu[ii]=1.0; }

/* Clean left and right edges. */
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=nc/2; } else { ecol=nc; }
  for (ii=0; ii<8; ++ii) {
    edge= nsx_find_real_image_row( 1, ii, nso );
    for (kk=0; kk<SPX[nso].numpro; ++kk) {
      jj = cnint(edge) + kk;
      pp = ii + (jj * nc);
      if ((valu[pp] > 1.05)||(valu[pp] < 0.95)) { valu[pp]=1.0; }
    }
  }
  for (ii=ecol-8; ii<ecol; ++ii) {
    edge= nsx_find_real_image_row( 1, ii, nso );
    for (kk=0; kk<SPX[nso].numpro; ++kk) {
      jj = cnint(edge) + kk;
      pp = ii + (jj * nc);
      if ((valu[pp] > 1.10)||(valu[pp] < 0.90)) { valu[pp]=1.0; }
    }
  }
}

/* Final. */
sprintf(wrd,"%s-comb.fits"  ,IA.root);  nsx_write_general_image( wrd, valu, nc, nr );      printf("Writing '%s'.\n",wrd);

free(Adivi1);
free(Adivi2);
free(Adivi3);
free(Bdivi1);
free(Bdivi2);
free(Bdivi3);
free(Bdivi4);
free(valu);
free(vwgt);
return;
}



/* ----------------------------------------------------------------------
  Build sky template (just add together -sp?.tbl data).  -tab 10aug2018
*/
void nsx_build_sky_template( char listfile[] )
{
/**/
char root[200];
char wrd[200];
char line[200];
/**/
int count,hicol[8],nso,col;
/**/
double sps[8][2100];
double spw[8][2100];
/**/
FILE *lsfu;
FILE *infu;
FILE *outfu;
/**/

/* Clear */
for (nso=3; nso<=7; ++nso) {
  for (col=0; col<2100; ++col) { sps[nso][col]=0.; spw[nso][col]=0.; }
  hicol[nso]=0;
}

/* Read files. */
count=0;
lsfu = fopen_read(listfile);
while (fgetline(root,lsfu)) {
  printf("Read '%s'.\n",root);
  for (nso=3; nso<=7; ++nso) {
    sprintf(wrd,"%s-sp%d.tbl",root,nso);
    infu = fopen_read(wrd);
    while (fgetline(line,infu)) { if (line[0] != '|') {
      col = GLV(line,6); 
      sps[nso][col] = sps[nso][col] + GLV(line,5);
      spw[nso][col] = spw[nso][col] + GLV(line,8);
      if (col > hicol[nso]) hicol[nso]=col;
    }} 
    fclose(infu);
  }
  ++count;
}
fclose(lsfu);
printf("Read %d files.\n",count);

/* Average. */
for (nso=3; nso<=7; ++nso) {
  for (col=0; col<2100; ++col) {
    sps[nso][col] = sps[nso][col] / (double)count;
    spw[nso][col] = spw[nso][col] / (double)count;
  }
}

/* Write. */
for (nso=3; nso<=7; ++nso) {
  sprintf(wrd,"sky_template%d.tbl",nso);
  outfu = fopen_write(wrd);
  fprintf(outfu,"| col | sky           | wave    |\n");
  for (col=0; col<=hicol[nso]; ++col) {
    fprintf(outfu," %5d %15.8e %9.3f\n",col,sps[nso][col],spw[nso][col]);
  }
  fclose(outfu);
}

return;
}



/* - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Process NIRES raw data files.
*/
int main(int argc, char *argv[])
{
/**/
char arg[202][200];
char wrd[300];
char wrd2[100];
char ffile[300];
char HOTroot[200];
char LOGfile[200];
char nsxdir[100];
char nsxout[100];
char userflat[200];
char sfx[200];
char listfile[200];
char nsxfile1[200];
char nsxfile2[200];
char UseAVP[100];
/**/
int checkflat,buildflat,boxrad,ii,jj,pixno,kk,lcB,ii1,ii2,nso,nxbk,nabk;
int nc,nr,numTYC,noobj,noback,noplot,autob,rbi,autox;
int ChangeAVP,TraceAVP,NoHotClean,NoFlat,NoClean,NoWindow,again,narg,ecol,mw;
int EarlyExit = 0;
/**/
long nbuffer;
/**/
float *scrimg;
float *darkimg;
/**/
double average,xsp1,xsp2,asp1,asp2;
double xbk1[MAXabk],xbk2[MAXabk],abk1[MAXabk],abk2[MAXabk];
double eperdn,objpos;
/**/
NLStype *NLS;
TYCtype *TYC;
/**/
SLTtype SLT[9];
SOPtype SOP1[9];
SOPtype SOP2[9];
SPXtype SPX[9];
SPXtype SPXB[9];
AVPtype AVP[9];
IMGtype IA;
IMGtype IB;
IMGtype IAB;
/**/
FILE *outfu0;
FILE *outfu1;
FILE *outfu2;
double xx[100],yy[100],pxsh,cen,edge,rowoff;
double asinc,asmax,as,sum,num,edge1,rb1,rb2,imgsum;
int iii,nn,jj1,jj2;
/**/

/* Copy arguments. */
cargcopy(argv, argc, &narg, arg);

/* Syntax. */
if (narg == 0) { nsx_syntax(); exit(0); }


/* Special hot pixel finder..  -tab 16aug2018 */
if (cfindarg(arg,&narg,"-HotPix",'s',wrd)) {
  nsx_HotPix();
  return(0);
}

/* Special low pixel finder..  -tab 21aug2018 */
if (cfindarg(arg,&narg,"-LowPix",'s',wrd)) {
  nsx_LowPix();
  return(0);
}


/* Clear Parameters. */
nsx_clear_SPX( SPX );
nsx_clear_SPX( SPXB);
nsx_clear_NCP( NCP );
nsx_clear_WSC( WSC );
nsx_clear_AVP( AVP );

/* Options. */
verbose  = cfindarg(arg,&narg,"-verbose" ,'s',wrd);

/* Which AVP file to use? */
strcpy(UseAVP,"");
if (cfindarg(arg,&narg,"avp=",':',wrd) == 1) { strcpy(UseAVP,wrd); }

/* Testing only */
objpos = 10.;
if (cfindarg(arg,&narg,"obj=",':',wrd) == 1) { objpos = GLV(wrd,1); }

/* Automatic object extraction. */
autox = cfindarg(arg,&narg,"-autox",'s',wrd);
if (autox == 0) autox = cfindarg(arg,&narg,"-auto",'s',wrd);

/* Other */
NoFlat    = cfindarg(arg,&narg,"-NoFlat",'s',wrd);
NoClean   = cfindarg(arg,&narg,"-NoClean",'s',wrd);
NoHotClean= cfindarg(arg,&narg,"-NoHotClean",'s',wrd);
rbi    = cfindarg(arg,&narg,"-rbi",'s',wrd);
autob  = cfindarg(arg,&narg,"-autob",'s',wrd);
noplot = cfindarg(arg,&narg,"-noplot",'s',wrd);
noback = cfindarg(arg,&narg,"-noback",'s',wrd);
noobj  = 1;

/* AVP recalibration. */
ChangeAVP = cfindarg(arg,&narg,"-newavp",'s',wrd);
if (ChangeAVP == 0) { ChangeAVP= cfindarg(arg,&narg,"-change",'s',wrd); }
TraceAVP  = cfindarg(arg,&narg,"-tavp",'s',wrd);
if (TraceAVP ==  0) { TraceAVP = cfindarg(arg,&narg,"-calavp",'s',wrd); }

/* eperdn guess. */
eperdn = 1.0; if (cfindarg(arg,&narg,"eperdn=" ,':',wrd) == 1) { eperdn=GLV(wrd,1); }
boxrad = 6;   if (cfindarg(arg,&narg,"boxrad=" ,':',wrd) == 1) { boxrad=GLV(wrd,1); }

/* Spectral extraction. */
xsp1=-999.; xsp2=-999.;
if (cfindarg(arg,&narg,"xsp=" ,':',wrd) == 1) { xsp1=GLV(wrd,1);  xsp2=GLV(wrd,2); }

/* Spectral extraction on arcsecond scale. */
asp1=-999.; asp2=-999.;
if (cfindarg(arg,&narg,"sp=",':',wrd) == 1) { asp1=GLV(wrd,1);  asp2=GLV(wrd,2); }


/* Background range(s). */
nxbk=0; again=1;
while (again) {
  if (cfindarg(arg,&narg,"xbk=" ,':',wrd) == 1) {  
    if (nxbk > 8) { printf("***error:bk=: too many background ranges given.\n"); exit(1); }
    xbk1[nxbk]=GLV(wrd,1);   xbk2[nxbk]=GLV(wrd,2); ++nxbk;  
    again=1;
  } else { again=0; }
}
nabk=0; again=1;
while (again) {
  if (cfindarg(arg,&narg,"bk=" ,':',wrd) == 1) {  
    if (nabk > 8) { printf("***error:abk=: too many background ranges given.\n"); exit(1); }
    abk1[nabk]=GLV(wrd,1);   abk2[nabk]=GLV(wrd,2); ++nabk;  
    again=1;
  } else { again=0; }
}

/* Check. */
if ((nxbk > 0)&&(nabk > 0)) {
  printf("***error: cannot use both bk= and xbk= in some command.\n"); exit(1);
}

/* Log file. */
strcpy(LOGfile,"");
if (cfindarg(arg,&narg,"log=" ,':',wrd) == 1) {  strcpy(LOGfile,wrd); }


/* Vega-like star for Atmospheric Correction. */
strcpy(HOTroot,"");
if (cfindarg(arg,&narg,"ac=" ,':',wrd) == 1) {  strcpy(HOTroot,wrd); }

/* User flat field. */
strcpy(userflat,"");

/* User suffix. */
strcpy(sfx,"");
if (cfindarg(arg,&narg,"sfx=",':',wrd) == 1) { strcpy(sfx,wrd); }


/* Special suffix. */
if (strcmp(sfx,"")==0) {
  if (TraceAVP) { strcat(sfx,"_tavp"); }
  if (ChangeAVP) { strcat(sfx,"_newavp"); }
}

/* List file. */
strcpy(listfile,"");
if (cfindarg(arg,&narg,"-list",'s',wrd) == 1) { strcpy(listfile,"stdout"); } else {
  if (cfindarg(arg,&narg,"list=",':',wrd) == 1) { strcpy(listfile,wrd); } 
}

/* Build sky template.. */
if (cfindarg(arg,&narg,"buildsky=",':',wrd) == 1) { 
  nsx_build_sky_template(wrd);
  return(0);
}

/* Build flat template.. */
buildflat = cfindarg(arg,&narg,"-buildflat",'s',wrd);
checkflat = cfindarg(arg,&narg,"-checkflat",'s',wrd);

/* Check NSXOUT environment variable. */
sprintf(wrd,"=%s=",getenv("NSXOUT"));
if (strcmp(wrd,"=(null)=") == 0) {
  if (VERB) printf("NOTE: No NSXOUT environment variable, writing output to default directory.\n");
  strcpy(nsxout,"");
} else {
  substrcpy_terminate(wrd,1,clc(wrd)-1,nsxout,0);
  if (FileExist(nsxout) == 0) {
    printf("***error: Cannot find NSXOUT directory '%s'.\n",nsxout); exit(1);
  }
  nsx_append_slash(nsxout);
  if (VERB) printf("NOTE: Output will be written to (NSXOUT) '%s'.\n",nsxout);
}

/* Check NSXDIR environment variable. */
sprintf(wrd,"=%s=",getenv("NSXDIR"));
if (strcmp(wrd,"=(null)=") == 0) {
  printf("***error: You must set the environment variable NSXDIR to the 'nsx' directory, e.g '/home/ptf/nsx' .\n");
  exit(1);
}

/* Get environment variable. */
strcpy(nsxdir,getenv("NSXDIR"));
if (FileExist(nsxdir) == 0) { printf("***error: The directory '%s', does not appear to exist.\n",nsxdir); exit(1); }
nsx_append_slash(nsxdir);
if (VERB) printf("NOTE: Found NSXDIR = '%s'.\n",nsxdir);

/* Set 'N2' for reference wavelength. */
nsx_set_N2_refwave();


/* Allocate. */
NLS = (NLStype *)calloc(MAXNLS,sizeof(NLStype));
TYC = (TYCtype *)calloc(MAXTYC,sizeof(TYCtype));

/* Read in nsx.tbl if it exists. */
nsx_load_NLS( nsxout, NLS );


/* Listing only? */
if (strcmp(listfile,"") != 0) {
  nsx_listing( listfile, arg[1], arg[2] );
  return(0);
}

/* Assumed size of FITS images. */
nc=nc_Nominal; nr=nr_Nominal; nbuffer = nc * nr;
scrimg  = (float *)calloc(((nc*nr)+1000),sizeof(float));
darkimg = (float *)calloc(((nc*nr)+1000),sizeof(float));

/* Read medianed dark, then shift it. */
if (READ_DARKIMG) {
  sprintf(ffile,"%scal/darkimg.fits",nsxdir);
  nsx_read_general_image( ffile, darkimg, nc, nr );
}

/* Check narg. */
if (narg > 2) { printf("***error: unknown options found (narg=%d).\n",narg); exit(1); }

/* Check first image file. */
strcpy(nsxfile1,arg[1]);
if (FileExist(nsxfile1) == 0) {
  strcat(nsxfile1,".fits");
  if (FileExist(nsxfile1) == 0) {
    printf("***error: Cannot find NIRES 'A' image file '%s'.\n",nsxfile1);
    exit(1);
  }
}

/* Argument 2?  (B file) */
if (narg == 2) {
  strcpy(nsxfile2,arg[2]);
  if (FileExist(nsxfile2) == 0) {
    strcat(nsxfile2,".fits");
    if (FileExist(nsxfile2) == 0) {
      printf("***error: Cannot find NIRES 'B' image file '%s'.\n",nsxfile2);
      exit(1);
    }
  }
} else {
  strcpy(nsxfile2,"");
}

/* Clear image structures. */
nsx_clear_image( &IA );
nsx_clear_image( &IB );
nsx_clear_image( &IAB );

/* Image files. */
strcpy(IA.file,nsxfile1); IA.X = FileExist(IA.file);
strcpy(IB.file,nsxfile2); IB.X = FileExist(IB.file);

/* Roots */
strcpy(IA.root,"");
ii1 = cindex_reverse(IA.file,"/");
ii2 = cindex(IA.file,".fits");
if (ii2-1 > ii1+1) { substrcpy_terminate(IA.file,ii1+1,ii2-1,IA.root,0); }
strcpy(IB.root,"");
if (IB.X) {
  ii1 = cindex_reverse(IB.file,"/");
  ii2 = cindex(IB.file,".fits");
  if (ii2-1 > ii1+1) { substrcpy_terminate(IB.file,ii1+1,ii2-1,IB.root,0); }
  printf("Found 2 image files: '%s' and '%s'.\n",IA.file,IB.file);
} else {
  printf("Found 1 image file: '%s' .\n",IA.file);
}

/* IAB image will be copied from IA, and new filename includes unique part of rootB.. */
if (IB.X) {
  strcpy(IAB.file,IA.file);
  kk=0; lcB=clc(IB.root);
  for (ii=0; ii<=lcB; ++ii) {
    if (IB.root[ii] == IA.root[ii]) { kk=ii+1; }
  }
  if (kk > lcB) kk=0;
  if (kk > lcB-3) kk=lcB-3;
  substrcpy_terminate(IB.root,kk,lcB,wrd2,0);
  sprintf(IAB.root,"%s-%s",IA.root,wrd2);
}

/* Log file? */
if (strcmp(LOGfile,"") == 0) {
  if (IB.X) { sprintf(LOGfile,"%s%s%s.log",nsxout,IAB.root,sfx); }
       else { sprintf(LOGfile,"%s%s%s.log",nsxout,IA.root,sfx);  }
}
printf("Writing to log file '%s'.\n",LOGfile);
logfu = fopen_write(LOGfile);
timegetstring(wrd); fprintf(logfu,"Start of log [%s].\n",wrd);
fprintf(logfu,"Running nsx version: %s .\n",NSXVERSION);

/* Read A image file. */
if (IB.X) { fprintf(logfu,"The A image:\n"); }
nsx_read_image( &IA, 1 );

/* Read B image file. */
if (IB.X) {
  fprintf(logfu,"The B image:\n");
  nsx_read_image( &IB, 1 );
}

/* Check. */
if (IB.X) {
  printf("IAB.root='%s'\n",IAB.root);
  fprintf(logfu,"The B image will be subtracted from the A image, and the A spectrum extracted.\n");
  fprintf(logfu,"The new root will be '%s'\n",IAB.root);
  if ((IB.nc != IA.nc)||(IB.nr != IA.nr)) {
    printf("***error: Dimensional mismatch between A (%d %d) and B (%d %d).\n",IA.nc,IA.nr,IB.nc,IB.nr);
    exit(1);
  }
}

/* Copy IA into IAB.. (read A again). */
if (IB.X) { nsx_read_image( &IAB, 0 ); }


/* Header check and set apparent zenith angle for differential atmospheric refraction. */
if (TRACE_TEST_ARCSEC > -9.) {
  nsx_header_check( IA );
}


/* Set AVP choice. */
if ((TraceAVP)||(ChangeAVP)||(Change_ASPP_in_AVP)||(Check_AVP_ASPP)) {
  if (strcmp(UseAVP,"") != 0) {
    printf("***error: 'UseAVP' must be blank with -tavp and -newavp options.\n");
    exit(1);
  }
} else {
  if (strcmp(UseAVP,"") != 0) {
    printf("Forcing use of '%s' date for AVP files.\n",UseAVP);
  } else {
    nsx_choose_best_AVP( IA, nsxdir, UseAVP );
  } 
} 
printf("Using '%s' date for AVP files.\n",UseAVP);
fprintf(logfu,"Using '%s' date for AVP files.\n",UseAVP);


/* Calibration Info. */
nsx_load_cal( nsxdir, SLT, SOP1, SOP2, AVP, TYC, &numTYC, UseAVP );


/* Check AVP files ASPP values. */
if (Check_AVP_ASPP) { nsx_Check_AVP_ASPP( AVP ); EarlyExit=1; }

/* Adjust ASPP in AVP calibration. */
if (Change_ASPP_in_AVP) { nsx_Change_ASPP_in_AVP( AVP, nsxdir, UseAVP ); EarlyExit=1; }

/* Change th AVP polynomials (take out 0.2 scaling).  [ NewAVP ]*/
if (ChangeAVP) { nsx_ChangeAVP( AVP, nsxdir ); EarlyExit=1; }


/* Early Exit. */
if (EarlyExit) {
  printf("EXITING EARLY...\n");
  timegetstring(wrd); fprintf(logfu,"End of log [%s].\n",wrd);
  fclose(logfu);
  return(0);
}


/* Shift, Flatten, Interpolate, Variance.
 * These are not done for NIRES.  No 'shift' is needed, the flat field is done later,
 * and the interpolation is done in the 'clean image' stage.
printf("Flatten, Interpolate, Variance.\n");
*/



/* SPECIAL: calibrate slant ... -tab 26apr2017  */
if (CALSLANT) {
  nsx_calslant( IA.image, IA.nc, IA.nr );
  free(darkimg); free(NLS); free(TYC);
  exit(0);
}


/* Stats. */
average = nsx_image_average( IA.image, IA.nc, IA.nr );
printf("NOTE: average=%f\n",average);


/* Wide profile mash to determine vertical shift of orders (SlitOffset). */
SlitOffset=0.;
nsx_wide_profile_mash( IA );


/* Set .numpro. NOTE: mw=upper-lower, but true numpro should be 1+upper-lower .. */
printf("Find .numpro ..\n");
for (nso=3; nso<=7; ++nso) {
  if (nso == 7) { ecol=IA.nc/2; } else { ecol=IA.nc; }
  mw = nsx_minwidth(ecol,nso);
  SPX[nso].numpro = 1 + mw;
  if (IB.X) SPXB[nso].numpro = 1 + mw;
}


/* Build flat field template.. */
if (buildflat) {
  nsx_clean_image( &IA, SPX, NoClean, NoHotClean, nsxdir );
  nsx_buildflat_template(IA,AVP,SPX,SOP1,SOP2,nsxdir);
  printf("EXITING EARLY (from buildflat ...)\n");
  timegetstring(wrd); fprintf(logfu,"End of log [%s].\n",wrd);
  fclose(logfu);
  return(0);
}


/* Co-add flats and check. */
if (checkflat) {
  nsx_checkflat_template(IA,AVP,SPX,SOP1,SOP2,nsxdir);
  printf("EXITING EARLY (from checkflat ...)\n");
  timegetstring(wrd); fprintf(logfu,"End of log [%s].\n",wrd);
  fclose(logfu);
  return(0);
}



/* Mash profile on arcsecond scale. */
printf("Profiles for '%s'.\n",IA.root);
nsx_arcsec_profile_mash( IA, SPX, AVP );
if (IB.X) {
  printf("Profiles for '%s'.\n",IB.root);
  nsx_arcsec_profile_mash( IB, SPXB, AVP );
}


/* SPECIAL: Calibration of point source curves as function of column. -tab 11dec2017 */
/* OBSOLETE OLD ROUTINES */
if (CALCURVE) {
  printf("Running CALCURVE..\n");
  nsx_calcurve( IA, SPX, AVP );
  free(darkimg); free(NLS); free(TYC);
  printf("End of CALCURVE..\n");
  exit(0);
}
/* SPECIAL: Calibration of AVP polynomials based on calcurve.dat data. -tab 13dec2017 */
/* OBSOLETE OLD ROUTINES */
if (CALCURVE2) {
  printf("Running CALCURVE2..\n");
  nsx_calcurve2( IA.nc );
  free(darkimg); free(NLS); free(TYC);
  printf("End of CALCURVE2..\n");
  exit(0);
}



/* Set object and background extraction windows based on command line parameters. */
NoWindow = nsx_set_extraction_windows(AVP,SPX,asp1,asp2,xsp1,xsp2,nabk,abk1,abk2,nxbk,xbk1,xbk2,noback,IA);
printf("NoWindow=%d\n",NoWindow);
if (NoWindow == 0) { nsx_echo_extraction_window( SPX, AVP, IA ); }

/* Find object and background automatically. */
if (((NoWindow)&&(autox))||(TraceAVP)) { 
  printf("Find object and background extraction windows automatically.\n");
  NoWindow = nsx_auto_window(IA,IB,IAB.root,SPX,SPXB,noback,nsxout);
  if (NoWindow == 0) { nsx_echo_extraction_window( SPX, AVP, IA ); }
}

/* Write profile tables. */
nsx_write_profiles( IA.root, AVP, SPX, nsxout, sfx, IA );
if (IB.X) { 
  nsx_write_profiles( IB.root, AVP, SPXB, nsxout, sfx, IA );
  nsx_write_AmB_profile( IAB.root, AVP, SPX, SPXB, nsxout, sfx, IAB );
}


/* SPECIAL: Profile centroid and RA,DEC position data. */
if (TESTPROCENT) {
  nsx_TestProCent( IA, AVP, SPX );
  exit(0);
}


/* Main object too faint? */
if ((autox)&&(ABS((SPX[0].sigs[0])) > 0.)&&(SPX[0].sigs[0] < MINOBJSIGS)) {
  printf("Main object too faint, no object selected automatically.\n");
  fprintf(logfu,"Main object too faint, no object selected automatically.\n");
  NoWindow = 1;
}

/* Echo */
printf("Object window: %6.2f to %6.2f arcseconds.  Object Sigmas: %7.2f .\n",
        SPX[0].asp1[0],SPX[0].asp2[0],SPX[0].sigs[0]);


/* Correct A image (un-slant and un-curve) (and clean CRs) (and do flat field division). */
nsx_correct_image( &IA, SOP1, SOP2, AVP, SPX, NoClean, NoHotClean, nsxdir, NoFlat );


/* Write. */
sprintf(wrd,"%s%s-corrected.fits",nsxout,IA.root);
nsx_write_general_image( wrd, IA.corimg,  IA.nc, IA.nr ); printf("Writing '%s'.\n",wrd);
sprintf(wrd,"%s%s-clnimg.fits",nsxout,IA.root);
nsx_write_general_image( wrd, IA.clnimg,  IA.nc, IA.nr ); printf("Writing '%s'.\n",wrd);



          /* Test position of trace along echelle orders. */
          if (TRACE_TEST_ARCSEC > -9.) {
          
          /* #@# */
          /* To verify DAR, remove it..
            IA.el = 0.;
          */
          /* #@# */
          
            outfu0 = fopen_write("trace.draw");
            fprintf(outfu0,"sci 3\n");
            for (nso=3; nso<=7; ++nso) {
              if (nso == 7) { ecol=IA.nc/2; } else { ecol=IA.nc; }
          
              sprintf(wrd,"trace_1cen%d_%s.dat",nso,IA.root); printf("Writing '%s'.\n",wrd);
              outfu1= fopen_write(wrd);
          
              sprintf(wrd,"trace_2cen%d.dat",nso); printf("Writing '%s'.\n",wrd);
              outfu2= fopen_write(wrd);
          
              for (ii=0; ii<ecol; ++ii) {
                edge  = nsx_find_real_image_row( 1, ii, nso );
                rowoff= nsx_AVPinv( AVP, nso, ii, TRACE_TEST_ARCSEC, IA );
          
          /* Centroid test 1. */
                asinc = ARCSEC_PER_PIXEL;
                asmax = cnint(( asinc * (double)SPX[3].numpro ));
                nn=0;
                for (as=0.; as<asmax; as=as+asinc) {
                  sum=0.; num=0.;
                  ii1 = ii-3; if (ii1 < 0) ii1=0;
                  ii2 = ii+3; if (ii2 > ecol-1) ii2=ecol-1;
                  for (iii=ii1; iii<=ii2; ++iii) {
                    edge1 = nsx_find_real_image_row( 1, iii, nso );
                    rb1 = edge1 + nsx_AVPinv( AVP, nso, iii, as, IA );
                    rb2 = edge1 + nsx_AVPinv( AVP, nso, iii, as+asinc, IA );
                    imgsum = nsx_fractional_pixel_rb( IA.nc, IA.clnimg, rb1, rb2, iii );
                    sum = sum + imgsum;
                    num = num + 1.;
                  }
                  xx[nn] = as + (asinc/2.);
                  yy[nn] = sum / num;
                  ++nn;
                }
                cen = nsx_centroid2( nn, xx, yy, TRACE_TEST_ARCSEC, (4. * ARCSEC_PER_PIXEL), 4, 1 );
          
          
                pxsh= (cen - TRACE_TEST_ARCSEC) / ARCSEC_PER_PIXEL;
          /*
                if (cindex(IA.root,"0054") > 0) { pxsh = pxsh + 0.3; }
          */
          
                fprintf(outfu1," %2d %4d %9.4f %9.4f %9.4f \n",nso,ii,TRACE_TEST_ARCSEC,cen,pxsh);
          
          /* Centroid test 2. */
                jj1 = cnint((edge+rowoff)) - 8;
                jj2 = cnint((edge+rowoff)) + 8;
                nn  = 0;
                for (jj=jj1; jj<jj2; ++jj) {
                  xx[nn] = (double)jj;
                  yy[nn] = IA.clnimg[(ii + (jj*IA.nc))];
                  ++nn;
                }
                cen = nsx_centroid2( nn, xx, yy, (edge+rowoff), 4., 4, 1 );
                fprintf(outfu2," %2d %4d %9.4f %9.4f %9.4f \n",nso,ii,(edge+rowoff),cen,cen-(edge+rowoff));
          
                fprintf(outfu0,"sym 3\n");
                fprintf(outfu0," %4d %9.4f \n",ii,edge);
                fprintf(outfu0,"plot\n");
          
                fprintf(outfu0,"sym 2\n");
                fprintf(outfu0," %4d %9.4f \n",ii,(edge+rowoff));
                fprintf(outfu0,"plot\n");
          
                fprintf(outfu0,"sym 4\n");
                fprintf(outfu0," %4d %9.4f \n",ii,cen);
                fprintf(outfu0,"plot\n");
          
                rowoff= nsx_AVPinv( AVP, nso, ii, 0., IA );
                fprintf(outfu0,"sym 5\n");
                fprintf(outfu0," %4d %9.4f \n",ii,(edge+rowoff));
                fprintf(outfu0,"plot\n");
          
              }
              fclose(outfu1);
              fclose(outfu2);
            }
            fclose(outfu0);
          
            printf("look at trace.draw.\n");
            EarlyExit=1;
          }


/* (re)Calibrate AVP polynomials. */
if (TraceAVP) { nsx_TraceAVP( IA, AVP, SPX, nsxdir ); EarlyExit=1; }


/* Early Exit. */
if (EarlyExit) {
  printf("EXITING EARLY...\n");
  timegetstring(wrd); fprintf(logfu,"End of log [%s].\n",wrd);
  fclose(logfu);
  return(0);
}

/* NOTE: Assuming eperdn=1.0 for now (which looks approx. correct). */

/* Copy to variance (use No Flat Division corrected image). */
for (ii=0; ii<IA.nc; ++ii) {
for (jj=0; jj<IA.nr; ++jj) {
  pixno = ii + (jj * IA.nc);
  IA.varimg[pixno] = IA.corimgNFD[pixno];
}}


/* Correct B image (un-slant and un-curve) (and clean CRs). */
if (IB.X) {
  nsx_correct_image( &IB, SOP1, SOP2, AVP, SPX, NoClean, NoHotClean, nsxdir, NoFlat );
  sprintf(wrd,"%s%s-corrected.fits",nsxout,IB.root);
  nsx_write_general_image( wrd, IB.corimg,  IB.nc, IB.nr ); printf("Writing '%s'.\n",wrd);


/* Subtract B from A. */
  for (ii=0; ii<IA.nc; ++ii) {
  for (jj=0; jj<IA.nr; ++jj) {
    pixno = ii + (jj * IA.nc);
    IAB.corimg[pixno] = IA.corimg[pixno] - IB.corimg[pixno];
    IAB.varimg[pixno] = IA.corimgNFD[pixno] + IB.corimgNFD[pixno];
  }}
  sprintf(wrd,"%s%s-corrected.fits",nsxout,IAB.root);
  nsx_write_general_image( wrd, IAB.corimg,  IB.nc, IB.nr ); printf("Writing '%s'.\n",wrd);

/* KVGC --  Subtract B from A for the uncorrected and store it  */
  for (ii=0; ii<IA.nc; ++ii) {
  for (jj=0; jj<IA.nr; ++jj) {
    pixno = ii + (jj * IA.nc);
    IAB.corimg[pixno] = IA.image[pixno] - IB.image[pixno];
  }}
  sprintf(wrd,"%s%s-corrected-KVGC.fits",nsxout,IAB.root);
  nsx_write_general_image( wrd, IAB.corimg,  IB.nc, IB.nr ); printf("KVGC -- Writing '%s'.\n",wrd);
}
/** */

/* Stop, if profiles only. */
if (NoWindow) {
  printf("Writing profiles only and corrected image, no extraction.\n");
  fprintf(logfu,"Writing profiles only and corrected image, no extraction.\n");
  free(darkimg);     darkimg    = NULL;
  free(NLS); free(TYC);
  for (nso=3; nso<=7; ++nso) { free(SLT[nso].image); }
  timegetstring(wrd); fprintf(logfu,"End of log [%s].\n",wrd);
  fclose(logfu);
  return(0);
}







/* Create background image and extract spectrum. */
if (IB.X) {

  nsx_create_background_image( IA, AVP, SPX );
  sprintf(wrd,"%s%s-sky.fits",nsxout,IAB.root ); nsx_write_general_image( wrd, IA.bckimg,  IAB.nc, IAB.nr );
  nsx_create_background_image( IAB, AVP, SPX );
  sprintf(wrd,"%s%s-bck.fits",nsxout,IAB.root); nsx_write_general_image( wrd, IAB.bckimg,  IAB.nc, IAB.nr );
  nsx_extract_spectrum( IAB, IA, AVP, SPX, nsxdir, nsxout );
  nsx_write_spectra( IAB.root, SPX, nsxdir, nsxout, sfx );

} else {

  nsx_create_background_image( IA, AVP, SPX );
  sprintf(wrd,"%s%s-bck.fits",nsxout,IA.root); nsx_write_general_image( wrd, IA.bckimg,  IA.nc, IA.nr );
  nsx_extract_spectrum( IA, IA, AVP, SPX, nsxdir, nsxout );
  nsx_write_spectra( IA.root, SPX, nsxdir, nsxout, sfx );
/*
  nsx_RMS_spectra( IA.root, SPX, nsxdir, nsxout, sfx );
*/
}



/* Free and close. */
free(darkimg);     darkimg    = NULL;
free(NLS); free(TYC);
for (nso=3; nso<=7; ++nso) { free(SLT[nso].image); }
timegetstring(wrd); fprintf(logfu,"End of log [%s].\n",wrd);
fclose(logfu);

return(0);
}

