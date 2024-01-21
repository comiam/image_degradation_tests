from super_image.models.a2n.configuration_a2n import A2nConfig
from super_image.models.a2n.modeling_a2n import A2nModel
from super_image.models.edsr.configuration_edsr import EdsrConfig
from super_image.models.edsr.modeling_edsr import EdsrModel
from super_image.models.msrn.configuration_msrn import MsrnConfig
from super_image.models.msrn.modeling_msrn import MsrnModel

AVAILABLE_MODELS = {
    EdsrModel: EdsrConfig(
        scale=4,
    ),
    MsrnModel: MsrnConfig(
        scale=4,
    ),
    A2nModel: A2nConfig(
        scale=4,
    ),
}
