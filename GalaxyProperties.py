class GalaxyProperties:
    def __init__(self,**kwargs):
        self.objid            = kwargs.get('objid')
        self.ra               = kwargs.get('ra')            ## ra in degrees
        self.dec              = kwargs.get('dec')           ## dec in degrees
        self.zs               = kwargs.get('zs')            ## redshift of the source galaxy
        self.zsNotes          = kwargs.get('zsNotes')
        self.zdef             = kwargs.get('zdef',[])       ## redshift of the deflector galaxy if known
        self.zdefNotes        = kwargs.get('zdefNotes')
        self.inst             = kwargs.get('inst')          ## Instrument used to observe target e.g ESI/NIRES/etc
        self.inst_pipeline    = kwargs.get('inst_pipeline') ## pipeline used to reduce the data
        self.inst_sigma       = kwargs.get('inst_sigma',1)  ## Instrument resolution sigma in km/s
        self.spec_filename    = kwargs.get('spec_filename') ## filename containing the fits file
        self.mass             = kwargs.get('mass',[])
        self.sfr              = kwargs.get('sfr',[])
        self.survey           = kwargs.get('survey')
        self.merger           = kwargs.get('merger')
        self.mergerNotes      = kwargs.get('mergerNotes')
        self.R                = kwargs.get('R')