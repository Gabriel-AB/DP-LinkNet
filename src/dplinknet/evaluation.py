from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from .utils import get_patches, stitch_together


class Binarization:
    TILE_SIZE: int = 256

    def __init__(self, net: type[nn.Module], device='cuda', quality: int = 2, hard: bool = True):
        self.hard = hard
        self.device = device
        self.quality = quality
        self.net = net().cuda() if device == 'cuda' else net()
        dev_ids = range(torch.cuda.device_count() if device == 'cuda' else 4)
        self.net = torch.nn.DataParallel(self.net, device_ids=dev_ids)
        for i in self.net.modules():
            if isinstance(i, nn.BatchNorm2d):
                i.eval()

    def __call__(self, image: torch.Tensor | np.ndarray | str | Path) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        return self.binarize(image)

    def cuda(self) -> 'Binarization':
        return self.__init__(self.net, device='cuda', quality=self.quality, hard=self.hard)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        # rotate 90 degree
        images = image[None]
        if self.quality >= 2:
            images = torch.cat([images, image.rot90()[None]])

            if self.quality >= 4:
                # flip y axis on 2 images 
                images = torch.cat([images, images.flip(1)])

                if self.quality >= 8:
                    # flip x axis on 4 images 
                    images = torch.cat([images, images.flip(2)])

        return images.moveaxis(3, 1).to(torch.float32) / 255.0 * 3.2 - 1.6

    def postprocess(self, images: torch.Tensor) -> torch.Tensor:

        if self.quality >= 8:
            # flip x axis and join 4 pairs of images
            images = images[:4] + images[4:].flip(2)
        
        if self.quality >= 4:
            # flip y axis and join 2 pairs of images
            images = images[:2] + images[2:].flip(1)

        if self.quality >= 2:
            # rotate -90 degree
            images = images[0] + images[1].rot90(-1)

        return images

    def binarize_subimage(self, image: torch.Tensor) -> torch.Tensor:
        batch = self.preprocess(image)
        outputs = self.net(batch).squeeze()
        output = self.postprocess(outputs)
        return output
    
    @torch.inference_mode()
    def binarize(self, image: torch.Tensor) -> torch.Tensor:
        
        locations, patches = get_patches(image, self.TILE_SIZE, self.TILE_SIZE)
        output = [self.binarize_subimage(pat) for pat in patches]
        output = stitch_together(locations, output, tuple(image.shape[0:2]), torch.device(self.device), self.TILE_SIZE, self.TILE_SIZE)

        if self.hard:
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
        return output

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
