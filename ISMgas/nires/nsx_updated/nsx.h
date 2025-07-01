/* nsx.h    General inclusions for nsx.c  */

/* Standard includes. */
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/dir.h>
#include <unistd.h>
#include <sys/time.h>

/* A few basic definition functions. */
#define SWAP(a,b) temp=(a);(a)=(b);(b)=temp;
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define ABS(x) ((x) < 0 ? (-1*x) : (x))

/* Julian date of zeroth 'modified night id' night for PTF. */
/* OLD OFFSET: #define MNID_ZERO 2454846.5 */
/* NEW OFFSET -tab 08sep2012 */
/* PTFJD == MNID == JD - MNID_ZERO */
#define MNID_ZERO 2454832.5

/* Palomar. */
#define PalomarLatitude   33.3561
#define PalomarLongitude 116.8639

/* Found from Google Maps.  -tab aug,oct2012 */
/* elevation from http://www.daftlogic.com/sandbox-google-maps-find-altitude.htm */
/* lat,long in decimal degrees.  elev in meters (+/- 5m) */
#define P48LAT     33.357323
#define P48LONG   116.859844
#define P48ELEV  1688.0
#define P60LAT     33.348356
#define P60LONG   116.859697
#define P60ELEV  1680.0
#define P200LAT    33.356303
#define P200LONG  116.864936
#define P200ELEV 1700.0

/* From: https://www.ifa.hawaii.edu/mko/coordinates.shtml */
#define KECK1LAT   19.82594655
#define KECK2LAT   19.82656052
#define KECK1LONG 155.47471851
#define KECK2LONG 155.47423408


/* Other functions that need to be defined external. */
extern double drand48(void);
extern double hypot(double x, double y);


/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. */

/* Rotation Matrices. */
struct RMstructure {
  double mat[3][3];        /* Sphere: Normal RA,Dec to prime system. */
  double inv[3][3];        /* Sphere: Prime system to normal RA,Dec. */
  double img[2][2];        /* Image: Normal x,y to "fixed-PA" x,y.  */
  double imginv[2][2];     /* Image: "Fixed-PA" x,y to normal x,y.  */
};
typedef struct RMstructure RMtype;


/* IPAC Table Column information (for a single column). */
struct ITCstructure {
  int ncol;       /* Number of columns: ITC[0].ncol .          */
  char name[30];  /* Column name.                              */
  int b1;         /* First boundary position.  (1=first character of a line) */
  int b2;         /* Second boundary position. (1=first character of a line) */
  char type;      /* Type of value in this column.             */
};
typedef struct ITCstructure ITCtype;


/* IFD structure (IPAC Float Data).   -tab 08jan2012 */
struct IFDstructure {
  char file[300];    /* Filename for IPAC table float data. */
  char **names;      /* Array of column names. */
  int  nc,nr;        /* Number of columns and rows. */
  float *img;        /* Float Data. */
};
typedef struct IFDstructure IFDtype;

/* IDD structure (IPAC Double Data).   -tab 08jan2012 */
struct IDDstructure {
  char file[300];    /* Filename for IPAC table double data. */
  char **names;      /* Array of column names. */
  int  nc,nr;        /* Number of columns and rows. */
  double *img;       /* Double Data. */
};
typedef struct IDDstructure IDDtype;


/* FWP structure (Fits World Pix). -tab 12jun2012 */
struct FWPstructure {
  double xrefval,yrefval;  
  double xrefpix,yrefpix;  
  double xinc,yinc,rot;
  char coordtype[80];
  int ncol,nrow;
};
typedef struct FWPstructure FWPtype;

