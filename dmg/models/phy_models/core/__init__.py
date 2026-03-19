from .alpine1 import (
    ALPINE1_PARAMS_BOUNDS,
    alpine1_step,
    create_initial_state as alpine1_init,
)
from .alpine2 import (
    ALPINE2_PARAMS_BOUNDS,
    alpine2_step,
    create_initial_state as alpine2_init,
)
from .australia import (
    AUSTRALIA_PARAMS_BOUNDS,
    australia_step,
    create_initial_state as australia_init,
)
from .collie1 import (
    COLLIE1_PARAMS_BOUNDS,
    collie1_step,
    create_initial_state as collie1_init,
)
from .collie2 import (
    COLLIE2_PARAMS_BOUNDS,
    collie2_step,
    create_initial_state as collie2_init,
)
from .collie3 import (
    COLLIE_PARAMS_BOUNDS,
    collie3_step,
    create_initial_state as collie3_init,
)
from .flexb import (
    FLEXB_PARAMS_BOUNDS,
    flexb_step,
    create_initial_state as flexb_init,
)
from .flexi import (
    FLEXI_PARAMS_BOUNDS,
    flexi_step,
    create_initial_state as flexi_init,
)
from .flexis import (
    FLEXIS_PARAMS_BOUNDS,
    flexis_step,
    create_initial_state as flexis_init,
)
from .gr4j import (
    GR4J_PARAMS_BOUNDS,
    gr4j_step,
    create_initial_state as gr4j_init,
)
from .gsfb import (
    GSFB_PARAMS_BOUNDS,
    gsfb_step,
    create_initial_state as gsfb_init,
)
from .hbv96 import (
    HBV96_PARAMS_BOUNDS,
    hbv96_step,
    create_initial_state as hbv96_init,
)
from .hillslope import (
    HILLSLOPE_PARAMS_BOUNDS,
    hillslope_step,
    create_initial_state as hillslope_init,
)
from .hymod import (
    HYMOD_PARAMS_BOUNDS,
    hymod_step,
    create_initial_state as hymod_init,
)
from .ihacres import (
    IHACRES_PARAMS_BOUNDS,
    ihacres_step,
    create_initial_state as ihacres_init,
)

# from .lascam import (
#     LASCAM_PARAMS_BOUNDS,
#     lascam_step,
#     create_initial_state as lascam_init,
# )
from .modhydrolog import (
    MODHYDROLOG_PARAMS_BOUNDS,
    modhydrolog_step,
    create_initial_state as modhydrolog_init,
)
from .mopex1 import (
    MOPEX1_PARAMS_BOUNDS,
    mopex1_step,
    create_initial_state as mopex1_init,
)
from .mopex2 import (
    MOPEX2_PARAMS_BOUNDS,
    mopex2_step,
    create_initial_state as mopex2_init,
)
from .mopex3 import (
    MOPEX3_PARAMS_BOUNDS,
    mopex3_step,
    create_initial_state as mopex3_init,
)
from .mopex4 import (
    MOPEX4_PARAMS_BOUNDS,
    mopex4_step,
    create_initial_state as mopex4_init,
)
from .mopex5 import (
    MOPEX5_PARAMS_BOUNDS,
    mopex5_step,
    create_initial_state as mopex5_init,
)
from .newzealand1 import (
    NEWZEALAND1_PARAMS_BOUNDS,
    newzealand1_step,
    create_initial_state as newzealand1_init,
)
from .newzealand2 import (
    NEWZEALAND2_PARAMS_BOUNDS,
    newzealand2_step,
    create_initial_state as newzealand2_init,
)
from .penman import (
    PENMAN_PARAMS_BOUNDS,
    penman_step,
    create_initial_state as penman_init,
)
from .plateau import (
    PLATEAU_PARAMS_BOUNDS,
    plateau_step,
    create_initial_state as plateau_init,
)

# from .sacramento import (
#     SACRAMENTO_PARAMS_BOUNDS,
#     sacramento_step,
#     create_initial_state as sacramento_init,
# )
from .simhyd import (
    SIMHYD_PARAMS_BOUNDS,
    simhyd_step,
    create_initial_state as simhyd_init,
)
from .smar import (
    SMAR_PARAMS_BOUNDS,
    smar_step,
    create_initial_state as smar_init,
)
from .susannah1 import (
    SUSANNAH1_PARAMS_BOUNDS,
    susannah1_step,
    create_initial_state as susannah1_init,
)
from .susannah2 import (
    SUSANNAH2_PARAMS_BOUNDS,
    susannah2_step,
    create_initial_state as susannah2_init,
)
from .tank import (
    TANK_PARAMS_BOUNDS,
    tank_step,
    create_initial_state as tank_init,
)
from .tcm import TCM_PARAMS_BOUNDS, tcm_step, create_initial_state as tcm_init
from .topmodel import (
    TOPMODEL_PARAMS_BOUNDS,
    topmodel_step,
    create_initial_state as topmodel_init,
)
from .us1 import US1_PARAMS_BOUNDS, us1_step, create_initial_state as us1_init
from .vic import VIC_PARAMS_BOUNDS, vic_step, create_initial_state as vic_init
from .wetland import (
    WETLAND_PARAMS_BOUNDS,
    wetland_step,
    create_initial_state as wetland_init,
)
from .xinanjiang import (
    XINANJIANG_PARAMS_BOUNDS,
    xinanjiang_step,
    create_initial_state as xinanjiang_init,
)

PARAM_INFO = {
    "alpine1": ALPINE1_PARAMS_BOUNDS,
    "alpine2": ALPINE2_PARAMS_BOUNDS,
    "australia": AUSTRALIA_PARAMS_BOUNDS,
    "collie1": COLLIE1_PARAMS_BOUNDS,
    "collie2": COLLIE2_PARAMS_BOUNDS,
    "collie3": COLLIE_PARAMS_BOUNDS,
    "flexb": FLEXB_PARAMS_BOUNDS,
    "flexi": FLEXI_PARAMS_BOUNDS,
    "flexis": FLEXIS_PARAMS_BOUNDS,
    "gr4j": GR4J_PARAMS_BOUNDS,
    "gsfb": GSFB_PARAMS_BOUNDS,
    "hbv96": HBV96_PARAMS_BOUNDS,
    "hillslope": HILLSLOPE_PARAMS_BOUNDS,
    "hymod": HYMOD_PARAMS_BOUNDS,
    "ihacres": IHACRES_PARAMS_BOUNDS,
    # "lascam": LASCAM_PARAMS_BOUNDS,
    "modhydrolog": MODHYDROLOG_PARAMS_BOUNDS,
    "mopex1": MOPEX1_PARAMS_BOUNDS,
    "mopex2": MOPEX2_PARAMS_BOUNDS,
    "mopex3": MOPEX3_PARAMS_BOUNDS,
    "mopex4": MOPEX4_PARAMS_BOUNDS,
    "mopex5": MOPEX5_PARAMS_BOUNDS,
    "newzealand1": NEWZEALAND1_PARAMS_BOUNDS,
    "newzealand2": NEWZEALAND2_PARAMS_BOUNDS,
    "penman": PENMAN_PARAMS_BOUNDS,
    "plateau": PLATEAU_PARAMS_BOUNDS,
    # "sacramento": SACRAMENTO_PARAMS_BOUNDS,
    "simhyd": SIMHYD_PARAMS_BOUNDS,
    "smar": SMAR_PARAMS_BOUNDS,
    "susannah1": SUSANNAH1_PARAMS_BOUNDS,
    "susannah2": SUSANNAH2_PARAMS_BOUNDS,
    "tank": TANK_PARAMS_BOUNDS,
    "tcm": TCM_PARAMS_BOUNDS,
    "topmodel": TOPMODEL_PARAMS_BOUNDS,
    "us1": US1_PARAMS_BOUNDS,
    "vic": VIC_PARAMS_BOUNDS,
    "wetland": WETLAND_PARAMS_BOUNDS,
    "xinanjiang": XINANJIANG_PARAMS_BOUNDS,
}

STFN_INFO = {
    "alpine1": alpine1_step,
    "alpine2": alpine2_step,
    "australia": australia_step,
    "collie1": collie1_step,
    "collie2": collie2_step,
    "collie3": collie3_step,
    "flexb": flexb_step,
    "flexi": flexi_step,
    "flexis": flexis_step,
    "gr4j": gr4j_step,
    "gsfb": gsfb_step,
    "hbv96": hbv96_step,
    "hillslope": hillslope_step,
    "hymod": hymod_step,
    "ihacres": ihacres_step,
    # "lascam": lascam_step,
    "modhydrolog": modhydrolog_step,
    "mopex1": mopex1_step,
    "mopex2": mopex2_step,
    "mopex3": mopex3_step,
    "mopex4": mopex4_step,
    "mopex5": mopex5_step,
    "newzealand1": newzealand1_step,
    "newzealand2": newzealand2_step,
    "penman": penman_step,
    "plateau": plateau_step,
    # "sacramento": sacramento_step,
    "simhyd": simhyd_step,
    "smar": smar_step,
    "susannah1": susannah1_step,
    "susannah2": susannah2_step,
    "tank": tank_step,
    "tcm": tcm_step,
    "topmodel": topmodel_step,
    "us1": us1_step,
    "vic": vic_step,
    "wetland": wetland_step,
    "xinanjiang": xinanjiang_step,
}

INIT_INFO = {
    "alpine1": alpine1_init,
    "alpine2": alpine2_init,
    "australia": australia_init,
    "collie1": collie1_init,
    "collie2": collie2_init,
    "collie3": collie3_init,
    "flexb": flexb_init,
    "flexi": flexi_init,
    "flexis": flexis_init,
    "gr4j": gr4j_init,
    "gsfb": gsfb_init,
    "hbv96": hbv96_init,
    "hillslope": hillslope_init,
    "hymod": hymod_init,
    "ihacres": ihacres_init,
    # "lascam": lascam_init,
    "modhydrolog": modhydrolog_init,
    "mopex1": mopex1_init,
    "mopex2": mopex2_init,
    "mopex3": mopex3_init,
    "mopex4": mopex4_init,
    "mopex5": mopex5_init,
    "newzealand1": newzealand1_init,
    "newzealand2": newzealand2_init,
    "penman": penman_init,
    "plateau": plateau_init,
    # "sacramento": sacramento_init,
    "simhyd": simhyd_init,
    "smar": smar_init,
    "susannah1": susannah1_init,
    "susannah2": susannah2_init,
    "tank": tank_init,
    "tcm": tcm_init,
    "topmodel": topmodel_init,
    "us1": us1_init,
    "vic": vic_init,
    "wetland": wetland_init,
    "xinanjiang": xinanjiang_init,
}


NPARAM_INFO = {
    "alpine1": 4,
    "alpine2": 6,
    "australia": 8,
    "collie1": 1,
    "collie2": 4,
    "collie3": 6,
    "flexb": 9,
    "flexi": 10,
    "flexis": 12,
    "gr4j": 4,
    "gsfb": 8,
    "hbv96": 15,
    "hillslope": 7,
    "hymod": 5,
    "ihacres": 6,
    # "lascam": 4,
    "modhydrolog": 15,
    "mopex1": 5,
    "mopex2": 7,
    "mopex3": 8,
    "mopex4": 10,
    "mopex5": 12,
    "newzealand1": 6,
    "newzealand2": 8,
    "penman": 4,
    "plateau": 8,
    "simhyd": 7,
    "smar": 8,
    "susannah1": 6,
    "susannah2": 6,
    "tank": 12,
    "tcm": 6,
    "topmodel": 7,
    "us1": 5,
    "vic": 10,
    "wetland": 2,
    "xinanjiang": 12,
}

STATE_INFO = {
    "alpine1": 1,
    "alpine2": 2,
    "australia": 3,
    "collie1": 1,
    "collie2": 1,
    "collie3": 2,
    "flexb": 4,
    "flexi": 5,
    "flexis": 5,
    "gr4j": 2,
    "gsfb": 4,
    "hbv96": 5,
    "hillslope": 4,
    "hymod": 5,
    "ihacres": 2,
    # "lascam": 4,
    "modhydrolog": 5,
    "mopex1": 4,
    "mopex2": 5,
    "mopex3": 5,
    "mopex4": 5,
    "mopex5": 5,
    "newzealand1": 3,
    "newzealand2": 2,
    "penman": 1,
    "plateau": 4,
    # "sacramento": 5,
    "simhyd": 3,
    "smar": 5,
    "susannah1": 5,
    "susannah2": 5,
    "tank": 6,
    "tcm": 1,
    "topmodel": 2,
    "us1": 4,
    "vic": 2,
    "wetland": 6,
    "xinanjiang": 4,
}

NUMBER_INFO = {
    "alpine1": 6,
    "alpine2": 12,
    "australia": 19,
    "collie1": 1,
    "collie2": 3,
    "collie3": 11,
    "flexb": 21,
    "flexi": 26,
    "flexis": 34,
    "gr4j": 7,
    "gsfb": 20,
    "hbv96": 37,
    "hillslope": 13,
    "hymod": 29,
    "ihacres": 5,
    # "lascam": lascam_init,
    "modhydrolog": 36,
    "mopex1": 24,
    "mopex2": 30,
    "mopex3": 31,
    "mopex4": 32,
    "mopex5": 35,
    "newzealand1": 4,
    "newzealand2": 16,
    "penman": 17,
    "plateau": 15,
    # "sacramento": sacramento_init,
    "simhyd": 18,
    "smar": 40,
    "susannah1": 9,
    "susannah2": 10,
    "tank": 27,
    "tcm": 25,
    "topmodel": 14,
    "us1": 8,
    "vic": 22,
    "wetland": 2,
    "xinanjiang": 28,
}

__all__ = ["PARAM_INFO", "STFN_INFO", "INIT_INFO", "STATE_INFO", "NUMBER_INFO"]
