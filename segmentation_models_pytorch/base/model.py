import numpy as np
import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        init.initialize_regression_head(self.regression_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)
        if self.adapter1 is not None:
            init.initialize_head(self.adapter1)
            init.initialize_head(self.adapter2)
            init.initialize_head(self.adapter3)
    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, batch):
        """training model with supervised data"""
        if len(batch) == 2:
            x, gts = batch
        else:
            x = batch
        self.check_input_shape(x) # [1,1,224,224]
        features = self.encoder(x) # [1,1,224,224]/[1,64,112,112]/[1,256,56,56]/[1,512,28,28]/[1,1024,14,14]/[1,2048,7,7]
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        regs = self.regression_head(decoder_output)
        if self.adapter1 is not None:
            feature2 = self.adapter1(features[2])
            feature3 = self.adapter2(features[3])
            feature4 = self.adapter3(features[4])
            return masks, regs, [feature2, feature3, feature4]
        else:
            return masks, regs, [None, None, None]


    # def forward(self, batch, masks=None, regs=None):
    #     """
    #     training one branch with supervised data
    #     training another branch with unlabeled data
    #     """
    #     x, gts = batch
    #
    #     self.check_input_shape(x)
    #
    #     features = self.encoder(x)
    #     decoder_output = self.decoder(*features)
    #
    #     label_indexs = torch.nonzero(gts == 1.0)[:, 0]
    #     unlabel_indexs = torch.nonzero(gts == 0.0)[:, 0]
    #     if len(label_indexs) != 0:
    #         decoder_output_label = decoder_output[label_indexs]
    #         masks = self.segmentation_head(decoder_output_label)
    #     if len(unlabel_indexs) != 0:
    #         decoder_output_unlabel = decoder_output[unlabel_indexs]
    #         regs = self.regression_head(decoder_output_unlabel)
    #
    #     if self.classification_head is not None:
    #         labels = self.classification_head(features[-1])
    #         return masks, labels
    #
    #     return masks, regs

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x

