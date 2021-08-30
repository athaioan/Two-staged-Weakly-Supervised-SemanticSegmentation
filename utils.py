import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
from numpy import loadtxt
import random
import matplotlib.pyplot as plt
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import skimage.measure
import random
import skimage.measure

class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1



def get_indices_of_pairs(radius, size):
## source: https://github.com/jiwoon-ahn/psa
    search_dist = []

    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius + 1, radius):
            if x * x + y * y < radius * radius:
                search_dist.append((y, x))

    radius_floor = radius - 1

    full_indices = np.reshape(np.arange(0, size[0]*size[1], dtype=np.int64),
                                   (size[0], size[1]))

    cropped_height = size[0] - radius_floor
    cropped_width = size[1] - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor],
                              [-1])

    indices_to_list = []

    for dy, dx in search_dist:
        indices_to = full_indices[dy:dy + cropped_height,
                     radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_to_list.append(indices_to)

    concat_indices_to = np.concatenate(indices_to_list, axis=0)

    return indices_from, concat_indices_to


def _fast_hist(label_true, label_pred, n_class):
    # source https://github.com/Juliachang/SC-CAM
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    # https://github.com/Juliachang/SC-CAM
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }




def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
## source: https://github.com/jiwoon-ahn/psa

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def random_crop(img, cropsize):
    # source https://github.com/jiwoon-ahn/psa.
    h, w, c = img.shape

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    container = np.zeros((cropsize, cropsize, img.shape[-1]), np.float32)
    container[cont_top:cont_top + ch, cont_left:cont_left + cw] = \
        img[img_top:img_top + ch, img_left:img_left + cw]

    return container



def AvgPool2d(img,ksize):
    ## source: https://github.com/jiwoon-ahn/psa

    return skimage.measure.block_reduce(img, (ksize, ksize, 1), np.mean)



def ExtractAffinityLabelInRadius(label, cropsize, radius=5):


        search_dist = []

        for x in range(1, radius):
            search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius + 1, radius):
                if x * x + y * y < radius * radius:
                    search_dist.append((y, x))

        radius_floor = radius - 1

        crop_height = cropsize - radius_floor
        crop_width = cropsize - 2 * radius_floor


        labels_from = label[:-radius_floor, radius_floor:-radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in search_dist:
            labels_to = label[dy:dy+crop_height, radius_floor+dx:radius_floor+dx+crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label).cuda(), torch.from_numpy(fg_pos_affinity_label).cuda(), torch.from_numpy(neg_affinity_label).cuda()




def resize_image(image, min_dim=None, max_dim=None, padding=False):
    ## modified function from https://github.com/matterport/Mask_RCNN

    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        # if round(image_max * scale) > max_dim:
        scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = np.array(Image.fromarray(image).resize((round(w * scale), round(h * scale))))

    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)

    return np.asarray(image,np.float32), window, scale, padding


# def HWC_to_CHW(img):
#     return np.transpose(img, (2, 0, 1))

class VOC2012Dataset(Dataset):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_names, labels_dict, voc12_img_folder, input_dim, transform=None):

        self.labels_dict = np.load(labels_dict, allow_pickle=True).item()
        self.transform = transform
        self.input_dim = input_dim


        with open(img_names) as file:
            self.img_paths = [voc12_img_folder + l.rstrip("\n")+".jpg" for l in file]

        self.pretrained_mean = np.asarray([123.68, 116.779, 103.939], np.float32) ## mean pixel value of the dataset that our pretrained have been trained with (Imagenet)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        current_path = self.img_paths[idx]

        img = PIL.Image.open(current_path).convert("RGB")
        if self.transform:
            img = self.transform(img) ## color jitering
            img = np.asarray(img)
            ### random horizontal flip with a probability of 0.5
            if np.random.random() > 0.5:
                img = np.flip(img, axis=1)
        else:
            img = np.asarray(img)

        ### resizing and padding the image to fix dimensions inputs
        orginal_shape = np.shape(img)

        img, window, _, _ = resize_image(img, min_dim=None,  max_dim=self.input_dim, padding=True)
        img = img - self.pretrained_mean
        imgBGR = img.copy()
        imgBGR[:, :, 0] = img[:, :, 2]
        imgBGR[:, :, 2] = img[:, :, 0]

        img = imgBGR.transpose(2, 0, 1)  # color channel in the first dim
        img = torch.from_numpy(img).cuda()

        img_key = current_path.split("/")[-1][:-4]
        label = torch.from_numpy(self.labels_dict[img_key]).cuda()

        return current_path, img, label, window, orginal_shape


class VOC2012DatasetCAM(Dataset):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_names, labels_dict, voc12_img_folder, input_dim):

        self.labels_dict = np.load(labels_dict, allow_pickle=True).item()
        self.input_dim = input_dim

        with open(img_names) as file:
            self.img_paths = [voc12_img_folder + l.rstrip("\n")+".jpg" for l in file]

        self.pretrained_mean = np.asarray([123.68, 116.779, 103.939], np.float32) ## mean pixel value of the dataset that our pretrained have been trained with (Imagenet)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        current_path = self.img_paths[idx]

        imgs = []
        windows = []

        for scale in [1]:#, 0.5, 1.5, 2]:
            for flip in [False, True]:
                img = PIL.Image.open(current_path).convert("RGB")
                img = np.asarray(img)


                original_img = img
                if flip:
                    img = np.flip(img, axis=1)

                ### resizing and padding the image to fix dimensions inputs
                orginal_shape = np.shape(img)
                img, window, _, _ = resize_image(img, min_dim=None,  max_dim=np.int(self.input_dim*scale), padding=True)
                img = img - self.pretrained_mean
                imgBGR = img.copy()
                imgBGR[:,:,0] = img[:,:,2]
                imgBGR[:,:,2] = img[:,:,0]


                img = imgBGR.transpose(2, 0, 1) #color channel in the first dim
                img = torch.from_numpy(img).cuda()

                img_key = current_path.split("/")[-1][:-4]
                label = torch.from_numpy(self.labels_dict[img_key]).cuda()

                imgs.append(img)
                windows.append(window)

        return current_path, imgs, label, windows, orginal_shape, original_img


class VOC2012Dataset_subcategory(Dataset):
    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_names, labels_dict, sublabels_dict, voc12_img_folder, input_dim, transform=None):

        self.labels_dict = np.load(labels_dict, allow_pickle=True).item()
        self.sublabels_dict = np.load(sublabels_dict, allow_pickle=True).item()

        self.transform = transform
        self.input_dim = input_dim

        with open(img_names) as file:
            self.img_paths = [voc12_img_folder + l.rstrip("\n") + ".jpg" for l in file]

        self.pretrained_mean = np.asarray([123.68, 116.779, 103.939],
                                          np.float32)  ## mean pixel value of the dataset that our pretrained have been trained with (Imagenet)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        current_path = self.img_paths[idx]

        img = PIL.Image.open(current_path).convert("RGB")
        if self.transform:
            img = self.transform(img)  ## color jitering
            img = np.asarray(img)
            ### random horizontal flip with a probability of 0.5
            if np.random.random() > 0.5:
                img = np.flip(img, axis=1)
        else:
            img = np.asarray(img)

        ### resizing and padding the image to fix dimensions inputs
        orginal_shape = np.shape(img)

        img, window, _, _ = resize_image(img, min_dim=None, max_dim=self.input_dim, padding=True)
        img = img - self.pretrained_mean
        imgBGR = img.copy()
        imgBGR[:, :, 0] = img[:, :, 2]
        imgBGR[:, :, 2] = img[:, :, 0]

        img = imgBGR.transpose(2, 0, 1)  # color channel in the first dim
        img = torch.from_numpy(img).cuda()

        img_key = current_path.split("/")[-1][:-4]
        label = torch.from_numpy(self.labels_dict[img_key]).cuda()
        sublabel = torch.from_numpy(self.sublabels_dict[img_key]).cuda()


        return current_path, img, label, sublabel, window, orginal_shape


class VOC2012Dataset_stage3(Dataset):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_names, cam_folder, labels_dict, voc12_img_folder, input_dim, transform=None):

        self.labels_dict = np.load(labels_dict, allow_pickle=True).item()
        self.transform = transform
        self.input_dim = input_dim
        self.cam_folder = cam_folder


        with open(img_names) as file:
            self.img_paths = [voc12_img_folder + l.rstrip("\n")+".jpg" for l in file]

        self.pretrained_mean = np.asarray([123.68, 116.779, 103.939], np.float32) ## mean pixel value of the dataset that our pretrained have been trained with (Imagenet)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        current_path = self.img_paths[idx]

        img_key = current_path.split("/")[-1].split(".")[0]
        img = PIL.Image.open(current_path).convert("RGB")

        cam = np.load( self.cam_folder+img_key+".npy")
        cam = np.amax(cam,axis=0)
        cam = 1-np.exp(-cam/(2*np.mean(cam)**2))

        ## transforming into the input img domain
        cam *=255
        cam = np.repeat(cam[None, ...], 3, axis=0).transpose(1,2,0)

        if self.transform:
            img = self.transform(img) ## color jitering
            img = np.asarray(img)
            ### random horizontal flip with a probability of 0.5
            if np.random.random() > 0.5:
                img = np.flip(img, axis=1)
                cam = np.flip(cam, axis=1)

        else:
            img = np.asarray(img)

        ### resizing and padding the image to fix dimensions inputs
        orginal_shape = np.shape(img)

        img, window, _, _ = resize_image(img, min_dim=None,  max_dim=self.input_dim, padding=True)
        cam, _, _, _ = resize_image(np.uint8(cam), min_dim=None,  max_dim=self.input_dim, padding=True)

        img = img - self.pretrained_mean
        cam = cam - self.pretrained_mean

        imgBGR = img.copy()
        imgBGR[:, :, 0] = img[:, :, 2]
        imgBGR[:, :, 2] = img[:, :, 0]
        camBGR = cam.copy()
        camBGR[:, :, 0] = cam[:, :, 2]
        camBGR[:, :, 2] = cam[:, :, 0]


        img = imgBGR.transpose(2, 0, 1)  # color channel in the first dim
        cam = camBGR.transpose(2, 0, 1)  # color channel in the first dim

        img = torch.from_numpy(img).cuda()
        cam = torch.from_numpy(cam).cuda()

        label = torch.from_numpy(self.labels_dict[img_key]).cuda()

        return current_path, img,label, window, orginal_shape, cam


class VOC2012DatasetCAM_stage3(Dataset):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_names, cam_folder, labels_dict, voc12_img_folder, input_dim):

        self.labels_dict = np.load(labels_dict, allow_pickle=True).item()
        self.input_dim = input_dim
        self.cam_folder = cam_folder

        with open(img_names) as file:
            self.img_paths = [voc12_img_folder + l.rstrip("\n")+".jpg" for l in file]

        self.pretrained_mean = np.asarray([123.68, 116.779, 103.939], np.float32) ## mean pixel value of the dataset that our pretrained have been trained with (Imagenet)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        current_path = self.img_paths[idx]

        imgs = []
        cams = []
        windows = []

        for scale in [1]:#, 0.5, 1.5, 2]:
            for flip in [False, True]:
                img = PIL.Image.open(current_path).convert("RGB")
                img = np.asarray(img)

                img_key = current_path.split("/")[-1].split(".")[0]

                cam = np.load(self.cam_folder + img_key + ".npy")
                cam = np.amax(cam, axis=0)
                cam = 1 - np.exp(-cam / (2 * np.mean(cam) ** 2))

                ## transforming into the input img domain
                cam *= 255
                cam = np.repeat(cam[None, ...], 3, axis=0).transpose(1, 2, 0)


                original_img = img
                if flip:
                    img = np.flip(img, axis=1)
                    cam = np.flip(cam, axis=1)


                ### resizing and padding the image to fix dimensions inputs
                orginal_shape = np.shape(img)
                cam, _, _, _ = resize_image(np.uint8(cam), min_dim=None,  max_dim=np.int(self.input_dim*scale), padding=True)
                img, window, _, _ = resize_image(img, min_dim=None,  max_dim=np.int(self.input_dim*scale), padding=True)
                img = img - self.pretrained_mean
                cam = cam - self.pretrained_mean



                imgBGR = img.copy()
                imgBGR[:,:,0] = img[:,:,2]
                imgBGR[:,:,2] = img[:,:,0]

                camBGR = cam.copy()
                camBGR[:, :, 0] = cam[:, :, 2]
                camBGR[:, :, 2] = cam[:, :, 0]

                img = imgBGR.transpose(2, 0, 1) #color channel in the first dim
                img = torch.from_numpy(img).cuda()

                cam = camBGR.transpose(2, 0, 1) #color channel in the first dim
                cam = torch.from_numpy(cam).cuda()

                label = torch.from_numpy(self.labels_dict[img_key]).cuda()

                imgs.append(img)
                cams.append(cam)

                windows.append(window)

        return current_path, imgs, label, windows, orginal_shape, original_img, cams


class VOC2012DatasetAffinity(Dataset):
    # source: https://github.com/jiwoon-ahn/psa

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_names, voc12_img_folder, input_dim, label_la_dir, label_ha_dir, radius,transform=None):

        self.transform = transform
        self.input_dim = input_dim
        self.label_la_dir = label_la_dir
        self.label_ha_dir = label_ha_dir
        self.radius = radius


        with open(img_names) as file:
            self.img_paths = [voc12_img_folder + l.rstrip("\n")+".jpg" for l in file]

        self.pretrained_mean = np.asarray([123.68, 116.779, 103.939], np.float32) ## mean pixel value of the dataset that our pretrained have been trained with (Imagenet)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        current_path = self.img_paths[idx]

        img_key = current_path.split("/")[-1].split(".")[0]

        img = PIL.Image.open(current_path).convert("RGB")

        low_a = np.load(self.label_la_dir + img_key +".npy").item()
        high_a = np.load(self.label_ha_dir + img_key +".npy").item()

        label = np.array(list(low_a.values()) + list(high_a.values()))
        label = np.transpose(label, (1, 2, 0))

        if self.transform:
            img = self.transform(img) ## color jitering
            img = np.asarray(img)
            ### random horizontal flip with a probability of 0.5
            if np.random.random() > 0.5:
                img = np.flip(img, axis=1)
                label = np.flip(label, axis=1)
        else:
            img = np.asarray(img)

        ### resizing and padding the image to fix dimensions inputs
        img_and_label = np.concatenate((img, label), axis=-1)
        img_and_label = random_crop(img_and_label,self.input_dim)

        img = img_and_label[..., :3]
        label = img_and_label[..., 3:]

        img = img - self.pretrained_mean
        imgBGR = img.copy()
        imgBGR[:, :, 0] = img[:, :, 2]
        imgBGR[:, :, 2] = img[:, :, 0]
        img = imgBGR.transpose(2, 0, 1)  # color channel in the first dim
        img = torch.from_numpy(img).cuda()

        label = AvgPool2d(label, 8)

        no_score_region = np.max(label, -1) < 1e-5 ## regios with significant low activation
        label_la, label_ha = np.array_split(label, 2, axis=-1)
        label_la = np.argmax(label_la, axis=-1).astype(np.uint8) ## category of each pixel
        label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8) ## category of each pixel
        label = label_la.copy()
        label[label_la == 0] = 255 ## neutral category ## non so strong indication of BG category
        label[label_ha == 0] = 0 ## background category ## strong indication of BG category
        label[no_score_region] = 255 ## neutral category ## these pixels have very low activation value even for the dominant category
                                        ## and thus are considered to be neutral

        label = ExtractAffinityLabelInRadius(label, self.input_dim//8, self.radius)

        return img, label

class VOC2012Dataset_cam_affinity(Dataset):

    # source: https://github.com/jiwoon-ahn/psa

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_names, labels_dict, voc12_img_folder, input_dim, transform=None):

        self.labels_dict = np.load(labels_dict, allow_pickle=True).item()
        self.transform = transform
        self.input_dim = input_dim


        with open(img_names) as file:
            self.img_paths = [voc12_img_folder + l.rstrip("\n")+".jpg" for l in file]

        self.pretrained_mean = np.asarray([123.68, 116.779, 103.939], np.float32) ## mean pixel value of the dataset that our pretrained have been trained with (Imagenet)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        current_path = self.img_paths[idx]

        img = PIL.Image.open(current_path).convert("RGB")
        if self.transform:
            img = self.transform(img) ## color jitering
            img = np.asarray(img)
            ### random horizontal flip with a probability of 0.5
            if np.random.random() > 0.5:
                img = np.flip(img, axis=1)
        else:
            img = np.asarray(img)

        ### resizing and padding the image to fix dimensions inputs
        orginal_shape = np.shape(img)

        # img, window, _, _ = resize_image(img, min_dim=None,  max_dim=self.input_dim, padding=True)
        img = img - self.pretrained_mean
        imgBGR = img.copy()
        imgBGR[:, :, 0] = img[:, :, 2]
        imgBGR[:, :, 2] = img[:, :, 0]

        img = imgBGR.transpose(2, 0, 1)  # color channel in the first dim
        img = torch.from_numpy(img).cuda()

        img_key = current_path.split("/")[-1][:-4]
        label = torch.from_numpy(self.labels_dict[img_key]).cuda()

        return current_path, img, label, orginal_shape


