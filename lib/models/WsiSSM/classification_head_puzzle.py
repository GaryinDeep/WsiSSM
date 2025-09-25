from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from . import slide_encoder


def reshape_input(imgs, coords, pad_mask=None):
    if len(imgs.shape) == 4:
        imgs = imgs.squeeze(0)
    if len(coords.shape) == 4:
        coords = coords.squeeze(0)
    if pad_mask is not None:
        if len(pad_mask.shape) != 2:
            pad_mask = pad_mask.squeeze(0)
    return imgs, coords, pad_mask


class ClassificationHead(nn.Module):
    """
    The classification head for the slide encoder

    Arguments:
    ----------
    input_dim: int
        The input dimension of the slide encoder
    latent_dim: int
        The latent dimension of the slide encoder
    feat_layer: str
        The layers from which embeddings are fed to the classifier, e.g., 5-11 for taking out the 5th and 11th layers
    n_classes: int
        The number of classes
    model_arch: str
        The architecture of the slide encoder
    pretrained: str
        The path to the pretrained slide encoder
    freeze: bool
        Whether to freeze the pretrained model
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        feat_layer,
        n_classes=2,
        model_arch="slide_enc12l768d",
        pretrained=None,
        freeze=False,
        **kwargs,
    ):
        super(ClassificationHead, self).__init__()

        # setup the slide encoder
        self.feat_layer = [eval(x) for x in feat_layer.split("-")]
        self.feat_dim = len(self.feat_layer) * latent_dim
        self.slide_encoder = slide_encoder.create_model(pretrained, model_arch, in_chans=input_dim, **kwargs)

        # whether to freeze the pretrained model
        if freeze:
            print("Freezing Pretrained model")
            for name, param in self.slide_encoder.named_parameters():
                param.requires_grad = False
            print("Done")
        # setup the classifier
        # self.classifier = nn.Sequential(*[nn.Linear(self.feat_dim, n_classes)])
        self.classifier = nn.Conv1d(in_channels=self.feat_dim, out_channels= n_classes, kernel_size=1, bias=False)

    def forward(self, images: torch.Tensor, coords: torch.Tensor, with_cam=True) -> torch.Tensor:
        """
        Arguments:
        ----------
        images: torch.Tensor
            The input images with shape [B, L, D]
        coords: torch.Tensor
            The input coordinates with shape [B, L, 2]
        """
        # inputs: [B, L, D]
        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        assert len(images.shape) == 3
        # forward slide encoder
        img_enc = self.slide_encoder.forward(images, coords, all_layer_embed=True, return_encoder_feature=True) # (n_layer,B,L,D)
        img_enc = [img_enc[i] for i in self.feat_layer] # [n,(B,L,D)]
        img_enc = torch.cat(img_enc, dim=-1) # (B,L,n*D)
        # classifier
        hidden_states = rearrange(img_enc, 'b l d -> b d l')
        if with_cam:
            features = self.classifier(hidden_states) # (B,2,L)
            logits = global_average_pooling_1d(features) # (B,2)
        else:
            features = global_average_pooling_1d(hidden_states, keepdims=True) # (B,1,2)
            logits = self.classifier(features).view(-1, self.num_classes) # # (B,2)
        return logits, features


def get_model(**kwargs):
    model = ClassificationHead(**kwargs)
    return model



def global_average_pooling_1d(x, keepdims=False):
    x = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        x = x.view(x.size(0), x.size(1), 1)
    return x


def make_cam(x, epsilon=1e-5):
    # relu(x) = max(x, 0)
    x = F.relu(x)  # （B,2,L）  
    b, c, l = x.size()

    max_value = x.max(axis=-1)[0].view((b, c, 1))
    return F.relu(x - epsilon) / (max_value + epsilon) # （B,2,L）  


def L1_Loss(A_tensors, B_tensors):
    return torch.abs(A_tensors - B_tensors)

def L2_Loss(A_tensors, B_tensors):
    return torch.pow(A_tensors - B_tensors, 2)


if __name__ == "__main__":
    data = torch.randn((1, 10000, 384)).cuda()
    model = ClassificationHead().cuda()
    logits, features = model(data)
    print(logits.shape)