import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
import torch.sparse as sparse

class VGG16(nn.Module):
    def __init__(self, n_classes, fc6_dilation=1):
        super(VGG16, self).__init__()

        self.train_history = {"loss": [],
                              "accuracy": []}
        self.val_history = {"loss": [],
                            "accuracy": []}
        self.min_val = np.inf
        self.n_classes = n_classes

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, 3, padding=fc6_dilation, dilation=fc6_dilation)
        self.drop6 = nn.Dropout2d() # 0.5 dropout

        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.drop7 = nn.Dropout2d() # 0.5 dropout


        self.fc8 = nn.Conv2d(1024, self.n_classes, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.from_scratch_layers = [self.fc8]

        return

    def feature_extractor(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        conv4 = x
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        conv5 = x

        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        x = F.relu(self.fc7(x))
        conv5fc = x

        return conv5fc

    def forward(self, x):

        x = self.feature_extractor(x)

        x = self.drop7(x)
        x = self.fc8(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, self.n_classes)

        return x


    def cam_output(self, x):

        x = self.feature_extractor(x)
        x = self.fc8(x) ## fc8 was set with bias=False and this is why applying the fc8 is like array multiplication as intructed in equation (1) of paper (1)
        x = F.relu(x)
        x = torch.sqrt(x) ## smoothed by square rooting to obtain a more uniform cam visualization
        return x

    def freeze_layers(self, frozen_stages):

        for layer in self.named_parameters():
            if "conv" in layer[0] and np.int(layer[0].split("conv")[-1][0]) in frozen_stages:
                layer[1].requires_grad = False

    def load_pretrained(self, pth_file):

        weights_dict = torch.load(pth_file)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if
                           k in model_dict and weights_dict[k].shape == model_dict[k].shape}
        #
        # no_pretrained_dict = {k: v for k, v in model_dict.items() if
        #                    not (k in weights_dict) or weights_dict[k].shape != model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        return


    def train_epoch(self, dataloader, optimizer, verbose=True):

        train_loss = 0
        train_accuracy = 0
        self.train()

        for index, data in enumerate(dataloader):

            img = data[1]
            cam = data[-1]
            label = data[2]

            x = self(img,cam)
            loss = F.multilabel_soft_margin_loss(x, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss += loss

            ### adding batch loss into the overall loss
            batch_accuracy = 1 - torch.mean((torch.sum(torch.abs(torch.sigmoid(x)>0.5 - label), axis=0) / x.shape[0]))
            train_accuracy += batch_accuracy

            if verbose:
                ### Printing epoch results
                print('Train Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Loss: {:.4f}\n'
                      'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch+1, self.epochs,
                                                      index + 1, len(dataloader),
                                                      loss.data.cpu().numpy(),
                                                      batch_accuracy.data.cpu().numpy()))


        self.train_history["loss"].append(train_loss / len(dataloader))
        self.train_history["accuracy"].append(train_accuracy / len(dataloader))

        return



    def val_epoch(self, dataloader, verbose=True):

           val_loss = 0
           val_accuracy = 0
           self.eval()

           with torch.no_grad():

               for index, data in enumerate(dataloader):

                   img = data[1]
                   label = data[2]

                   x = self(img)
                   loss = F.multilabel_soft_margin_loss(x, label)

                   ### adding batch loss into the overall loss
                   val_loss += loss

                   ### adding batch loss into the overall loss
                   batch_accuracy = 1 - torch.mean((torch.sum(torch.abs(torch.sigmoid(x)>0.5 - label), axis=0) / x.shape[0]))
                   val_accuracy += batch_accuracy

                   if verbose:
                       ### Printing epoch results
                       print('Val Epoch: {}/{}\n'
                             'Step: {}/{}\n'
                             'Batch ~ Loss: {:.4f}\n'
                             'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                                 index + 1, len(dataloader),
                                                                 loss.data.cpu().numpy(),
                                                                 batch_accuracy.data.cpu().numpy()))

               self.val_history["loss"].append(val_loss / len(dataloader))
               self.val_history["accuracy"].append(val_accuracy / len(dataloader))


           return


    def extract_cams(self, dataloader, cam_folder, low_a, high_a):

        if not os.path.exists(cam_folder):
            os.makedirs(cam_folder)


        with torch.no_grad():
            for index, data in enumerate(dataloader):
                print(str(index)+" / " + str(len(dataloader)))
                #current_path, imgs, label, windows, orginal_shape, original_img
                original_shape = data[4]
                label = data[2]

                imgs = data[1]
                windows = data[3]
                img_original = data[5][0].data.cpu().numpy()

                final_cam = np.zeros([self.n_classes, original_shape[0], original_shape[1]])
                final_cam_unlabeled = np.zeros([self.n_classes, original_shape[0], original_shape[1]])

                for index, img in enumerate(imgs):

                    window = windows[index]


                    x = self.cam_output(img)


                    x = F.upsample(x, [img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)[0]

                    ## removing the crop window
                    x = x[:, window[0]:window[2], window[1]:window[3]]

                    x = F.upsample(x.unsqueeze(0), [original_shape[0].data.cpu().numpy()[0], original_shape[1].data.cpu().numpy()[0]], mode='bilinear', align_corners=False)[0]

                    ## filter out non-existing classes
                    cam = x.cpu().numpy() * label.clone().view(self.n_classes, 1, 1).data.cpu().numpy()
                    cam_unlabeled = x.cpu().numpy()

                    if index % 2 == 1:
                        cam = np.flip(cam, axis=2)
                        cam_unlabeled = np.flip(cam_unlabeled, axis=2)

                    final_cam += cam
                    final_cam_unlabeled += cam_unlabeled

                ## normalizing final_cam
                denom = np.max(final_cam, (1, 2))
                denom_unlabeled = np.max(final_cam_unlabeled, (1, 2))

                ## when class does not exist then divide by one
                denom += 1 - (denom > 0)
                denom_unlabeled += 1 - (denom_unlabeled > 0)

                final_cam /= denom.reshape(self.n_classes, 1, 1)

                ########## savings cams as dict
                final_cam_dict = {}
                final_cam_dict_unlabeled = {}

                for i in range(self.n_classes):
                    final_cam_dict_unlabeled[i] = final_cam_unlabeled[i]
                    if label[0][i] == 1:
                        final_cam_dict[i] = final_cam[i]



                existing_cams = np.asarray(list(final_cam_dict.values()))
                existing_cams_unlabeled = np.asarray(list(final_cam_dict_unlabeled.values()))

                bg_score_low = np.power(1 - np.max(existing_cams, axis=0), low_a)
                bg_score_high = np.power(1 - np.max(existing_cams, axis=0), high_a)

                bg_score_low_cams = np.concatenate((np.expand_dims(bg_score_low, 0), existing_cams), axis=0)
                bg_score_high_cams = np.concatenate((np.expand_dims(bg_score_high, 0), existing_cams), axis=0)

                crf_score_low = crf_inference(img_original, bg_score_low_cams, labels=bg_score_low_cams.shape[0])
                crf_score_high = crf_inference(img_original, bg_score_high_cams, labels=bg_score_high_cams.shape[0])

                crf_low_dict = {}
                crf_high_dict = {}

                crf_low_dict[0] = crf_score_low[0]
                crf_high_dict[0] = crf_score_high[0]

                for i, key in enumerate(final_cam_dict.keys()):
                    # plus one to account for the added BG class
                    crf_low_dict[key+1] = crf_score_low[i+1]
                    crf_high_dict[key+1] = crf_score_high[i+1]

                if not os.path.exists(cam_folder+"low"):
                    os.makedirs(cam_folder+"low")

                if not os.path.exists(cam_folder+"high"):
                    os.makedirs(cam_folder+"high")

                if not os.path.exists(cam_folder + "default"):
                    os.makedirs(cam_folder + "default")


                np.save(cam_folder+"low/"+data[0][0].split("/")[-1][:-4]+".npy", crf_low_dict)
                np.save(cam_folder+"high/"+data[0][0].split("/")[-1][:-4]+".npy", crf_high_dict)
                np.save(cam_folder+"default/"+data[0][0].split("/")[-1][:-4]+".npy", existing_cams_unlabeled)



    def evaluate_cams(self,dataloader, cam_folder, gt_mask_folder):

        t_hold = 0.2
        c_num = np.zeros(21)
        c_denom = np.zeros(21)
        gt_masks = []
        preds = []


        for index, data in enumerate(dataloader):

            img = data[1]
            label = data[2]

            img_key = data[0][0].split("/")[-1].split(".")[0]
            # I = plt.imread("C:/Users/johny/Desktop/ProjectV2/VOCdevkit/VOC2012/JPEGImages/"+img_key+".jpg")
            ## loading generated cam
            # cam_low = np.load(cam_folder +"low/" + img_key + '.npy').item()
            cam_default = np.load(cam_folder +"default/" + img_key + '.npy')
            # cam_high = np.load(cam_folder +"high/" + img_key + '.npy').item()

            ## considering only the labeled cams
            cam_default = cam_default * label.clone().view(self.n_classes, 1, 1).data.cpu().numpy()

            ## normalizing final_cam
            denom = np.max(cam_default, (1, 2))

            ## when class does not exist then divide by one
            denom += 1 - (denom > 0)

            cam_default /= denom.reshape(self.n_classes, 1, 1)


            bg_score = np.expand_dims(np.ones_like(cam_default[0])*t_hold,0)
            pred = np.argmax(np.concatenate((bg_score, cam_default)), 0)



            ## loading ground truth annotated mask
            gt_mask = Image.open(gt_mask_folder + img_key + '.png')
            gt_mask = np.array(gt_mask)

            preds.append(pred)
            gt_masks.append(gt_mask)


        sc = scores(gt_masks, preds, self.n_classes+1)

        return sc

    def extract_sub_category(self,dataloader,sub_folder):

        features={}
        ids = {}

        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)


        for i in range(20):
            features[i] = []
            ids[i] =[]

        with torch.no_grad():
            for index, data in enumerate(dataloader):

                img = data[1]
                key = data[0][0].split("/")[-1].split(".")[0]
                label = data[2].data.cpu().numpy()
                x = self.feature_extractor(img)
                feat = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0).squeeze().data.cpu().numpy()
                feat /= np.linalg.norm(feat)

                id = np.where(label[0])[0]

                for i in id:
                    ids[i].append(key)
                    features[i].append(feat)

            np.save(sub_folder+"features.npy", features)
            np.save(sub_folder+"ids.npy", ids)


    def visualize_graph(self):

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")

        plt.plot(np.arange(len(self.train_history["loss"])), self.train_history["loss"], label="train")
        plt.plot(np.arange(len(self.val_history["loss"])), self.val_history["loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"loss.png")
        plt.close()


        ## Plotting accyracy
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Graph")

        plt.plot(np.arange(len(self.train_history["accuracy"])), self.train_history["accuracy"], label="train")
        plt.plot(np.arange(len(self.val_history["accuracy"])), self.val_history["accuracy"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"accuracy.png")
        plt.close()



    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)):

                if m.weight is not None and m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups



class VGG16_stage3(nn.Module):
    def __init__(self, n_classes, fc6_dilation=1):
        super(VGG16_stage3, self).__init__()

        self.train_history = {"loss": [],
                              "accuracy": []}
        self.val_history = {"loss": [],
                            "accuracy": []}
        self.min_val = np.inf
        self.n_classes = n_classes

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, 3, padding=fc6_dilation, dilation=fc6_dilation)
        self.drop6 = nn.Dropout2d() # 0.5 dropout

        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.drop7 = nn.Dropout2d() # 0.5 dropout


        self.fc8 = nn.Conv2d(2048, self.n_classes, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.from_scratch_layers = [self.fc8]

        return

    def feature_extractor(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        conv4 = x
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        conv5 = x

        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        x = F.relu(self.fc7(x))
        conv5fc = x

        return conv5fc

    def forward(self, x, x_cam):

        x1 = self.feature_extractor(x)
        x1 = self.drop7(x1)

        x2 = self.feature_extractor(x_cam)
        x2 = self.drop7(x2)

        x = torch.cat((x1, x2), 1)

        x = self.fc8(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, self.n_classes)

        return x


    def cam_output(self, x, x_cam):

        x1 = self.feature_extractor(x)
        x2 = self.feature_extractor(x_cam)

        x = torch.cat((x1, x2), 1)


        x = self.fc8(x) ## fc8 was set with bias=False and this is why applying the fc8 is like array multiplication as intructed in equation (1) of paper (1)
        x = F.relu(x)
        x = torch.sqrt(x) ## smoothed by square rooting to obtain a more uniform cam visualization
        return x

    def freeze_layers(self, frozen_stages):

        for layer in self.named_parameters():
            if "conv" in layer[0] and np.int(layer[0].split("conv")[-1][0]) in frozen_stages:
                layer[1].requires_grad = False

    def load_pretrained(self, pth_file):

        weights_dict = torch.load(pth_file)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if
                           k in model_dict and weights_dict[k].shape == model_dict[k].shape}
        #
        # no_pretrained_dict = {k: v for k, v in model_dict.items() if
        #                    not (k in weights_dict) or weights_dict[k].shape != model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        return


    def train_epoch(self, dataloader, optimizer, verbose=True):

        train_loss = 0
        train_accuracy = 0
        self.train()

        for index, data in enumerate(dataloader):

            img = data[1]
            cam = data[-1]
            label = data[2]

            x = self(img,cam)
            loss = F.multilabel_soft_margin_loss(x, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss += loss

            ### adding batch loss into the overall loss
            batch_accuracy = 1 - torch.mean((torch.sum(torch.abs(torch.sigmoid(x)>0.5 - label), axis=0) / x.shape[0]))
            train_accuracy += batch_accuracy

            if verbose:
                ### Printing epoch results
                print('Train Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Loss: {:.4f}\n'
                      'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch+1, self.epochs,
                                                      index + 1, len(dataloader),
                                                      loss.data.cpu().numpy(),
                                                      batch_accuracy.data.cpu().numpy()))


        self.train_history["loss"].append(train_loss / len(dataloader))
        self.train_history["accuracy"].append(train_accuracy / len(dataloader))

        return



    def val_epoch(self, dataloader, verbose=True):

           val_loss = 0
           val_accuracy = 0
           self.eval()

           with torch.no_grad():

               for index, data in enumerate(dataloader):

                   img = data[1]
                   cam = data[-1]
                   label = data[2]

                   x = self(img, cam)
                   loss = F.multilabel_soft_margin_loss(x, label)

                   ### adding batch loss into the overall loss
                   val_loss += loss

                   ### adding batch loss into the overall loss
                   batch_accuracy = 1 - torch.mean((torch.sum(torch.abs(torch.sigmoid(x)>0.5 - label), axis=0) / x.shape[0]))
                   val_accuracy += batch_accuracy

                   if verbose:
                       ### Printing epoch results
                       print('Val Epoch: {}/{}\n'
                             'Step: {}/{}\n'
                             'Batch ~ Loss: {:.4f}\n'
                             'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                                 index + 1, len(dataloader),
                                                                 loss.data.cpu().numpy(),
                                                                 batch_accuracy.data.cpu().numpy()))

               self.val_history["loss"].append(val_loss / len(dataloader))
               self.val_history["accuracy"].append(val_accuracy / len(dataloader))


           return


    def extract_cams(self, dataloader, cam_folder, low_a, high_a):

        if not os.path.exists(cam_folder):
            os.makedirs(cam_folder)


        with torch.no_grad():
            for index, data in enumerate(dataloader):
                print(str(index)+" / " + str(len(dataloader)))
                #current_path, imgs, label, windows, orginal_shape, original_img
                original_shape = data[4]
                label = data[2]

                imgs = data[1]
                cams = data[-1]
                windows = data[3]
                img_original = data[5][0].data.cpu().numpy()

                final_cam = np.zeros([self.n_classes, original_shape[0], original_shape[1]])
                final_cam_unlabeled = np.zeros([self.n_classes, original_shape[0], original_shape[1]])

                for index, img in enumerate(imgs):

                    window = windows[index]


                    x = self.cam_output(img, cams[index])


                    x = F.upsample(x, [img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)[0]

                    ## removing the crop window
                    x = x[:, window[0]:window[2], window[1]:window[3]]

                    x = F.upsample(x.unsqueeze(0), [original_shape[0].data.cpu().numpy()[0], original_shape[1].data.cpu().numpy()[0]], mode='bilinear', align_corners=False)[0]

                    ## filter out non-existing classes
                    cam = x.cpu().numpy() * label.clone().view(self.n_classes, 1, 1).data.cpu().numpy()
                    cam_unlabeled = x.cpu().numpy()

                    if index % 2 == 1:
                        cam = np.flip(cam, axis=2)
                        cam_unlabeled = np.flip(cam_unlabeled, axis=2)

                    final_cam += cam
                    final_cam_unlabeled += cam_unlabeled

                ## normalizing final_cam
                denom = np.max(final_cam, (1, 2))
                denom_unlabeled = np.max(final_cam_unlabeled, (1, 2))

                ## when class does not exist then divide by one
                denom += 1 - (denom > 0)
                denom_unlabeled += 1 - (denom_unlabeled > 0)

                final_cam /= denom.reshape(self.n_classes, 1, 1)

                ########## savings cams as dict
                final_cam_dict = {}
                final_cam_dict_unlabeled = {}

                for i in range(self.n_classes):
                    final_cam_dict[i] = final_cam[i]
                    final_cam_dict_unlabeled[i] = final_cam_unlabeled[i]

                existing_cams = np.asarray(list(final_cam_dict.values()))
                existing_cams_unlabeled = np.asarray(list(final_cam_dict_unlabeled.values()))

                # bg_score_low = np.power(1 - np.max(existing_cams, axis=0), low_a)
                # bg_score_high = np.power(1 - np.max(existing_cams, axis=0), high_a)
                #
                # bg_score_low_cams = np.concatenate((np.expand_dims(bg_score_low, 0), existing_cams), axis=0)
                # bg_score_high_cams = np.concatenate((np.expand_dims(bg_score_high, 0), existing_cams), axis=0)
                #
                # crf_score_low = crf_inference(img_original, bg_score_low_cams, labels=bg_score_low_cams.shape[0])
                # crf_score_high = crf_inference(img_original, bg_score_high_cams, labels=bg_score_high_cams.shape[0])
                #
                # crf_low_dict = {}
                # crf_high_dict = {}
                #
                # crf_low_dict[0] = crf_score_low[0]
                # crf_high_dict[0] = crf_score_high[0]
                #
                # for i in range(self.n_classes):
                #     crf_low_dict[i+1] = crf_score_low[i+1]
                #     crf_high_dict[i+1] = crf_score_high[i+1]
                #
                # if not os.path.exists(cam_folder+"low"):
                #     os.makedirs(cam_folder+"low")
                #
                # if not os.path.exists(cam_folder+"high"):
                #     os.makedirs(cam_folder+"high")

                if not os.path.exists(cam_folder + "default"):
                    os.makedirs(cam_folder + "default")


                # np.save(cam_folder+"low/"+data[0][0].split("/")[-1][:-4]+".npy", crf_low_dict)
                # np.save(cam_folder+"high/"+data[0][0].split("/")[-1][:-4]+".npy", crf_high_dict)
                np.save(cam_folder+"default/"+data[0][0].split("/")[-1][:-4]+".npy", existing_cams_unlabeled)



    def evaluate_cams(self,dataloader, cam_folder, gt_mask_folder):

        t_hold = 0.2
        c_num = np.zeros(21)
        c_denom = np.zeros(21)
        gt_masks = []
        preds = []


        for index, data in enumerate(dataloader):

            img = data[1]
            label = data[2]

            img_key = data[0][0].split("/")[-1].split(".")[0]

            ## loading generated cam
            # cam_low = np.load(cam_folder +"low/" + img_key + '.npy').item()
            cam_default = np.load(cam_folder +"default/" + img_key + '.npy')
            # cam_high = np.load(cam_folder +"high/" + img_key + '.npy').item()

            ## considering only the labeled cams
            cam_default = cam_default * label.clone().view(self.n_classes, 1, 1).data.cpu().numpy()

            ## normalizing final_cam
            denom = np.max(cam_default, (1, 2))

            ## when class does not exist then divide by one
            denom += 1 - (denom > 0)

            cam_default /= denom.reshape(self.n_classes, 1, 1)


            bg_score = np.expand_dims(np.ones_like(cam_default[0])*t_hold,0)
            pred = np.argmax(np.concatenate((bg_score, cam_default)), 0)



            ## loading ground truth annotated mask
            gt_mask = Image.open(gt_mask_folder + img_key + '.png')
            gt_mask = np.array(gt_mask)

            preds.append(pred)
            gt_masks.append(gt_mask)


        sc = scores(gt_masks, preds, self.n_classes+1)

        return sc

    def extract_sub_category(self,dataloader,sub_folder):

        features={}
        ids = {}

        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)


        for i in range(20):
            features[i] = []
            ids[i] =[]

        with torch.no_grad():
            for index, data in enumerate(dataloader):

                img = data[1]
                key = data[0][0].split("/")[-1].split(".")[0]
                label = data[2].data.cpu().numpy()
                x = self.feature_extractor(img)
                feat = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0).squeeze().data.cpu().numpy()
                feat /= np.linalg.norm(feat)

                id = np.where(label[0])[0]

                for i in id:
                    ids[i].append(key)
                    features[i].append(feat)

            np.save(sub_folder+"features.npy", features)
            np.save(sub_folder+"ids.npy", ids)


    def visualize_graph(self):

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")

        plt.plot(np.arange(len(self.train_history["loss"])), self.train_history["loss"], label="train")
        plt.plot(np.arange(len(self.val_history["loss"])), self.val_history["loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"loss.png")
        plt.close()


        ## Plotting accyracy
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Graph")

        plt.plot(np.arange(len(self.train_history["accuracy"])), self.train_history["accuracy"], label="train")
        plt.plot(np.arange(len(self.val_history["accuracy"])), self.val_history["accuracy"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"accuracy.png")
        plt.close()



    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)):

                if m.weight is not None and m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups



class VGG16_sub(nn.Module):
    def __init__(self, n_classes, n_subclasses, fc6_dilation=1):
        super(VGG16_sub, self).__init__()

        self.train_history = {"loss": [],
                              "cls_loss": [],
                              "subcls_loss": [],
                              "cls_accuracy": [],
                              "subcls_accuracy": []}

        self.val_history = {"loss": [],
                              "cls_loss": [],
                              "subcls_loss": [],
                              "cls_accuracy": [],
                              "subcls_accuracy": []}



        self.min_val = np.inf
        self.n_classes = n_classes
        self.sub_classes = n_subclasses

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, 3, padding=fc6_dilation, dilation=fc6_dilation)
        self.drop6 = nn.Dropout2d() # 0.5 dropout

        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.drop7 = nn.Dropout2d() # 0.5 dropout


        self.fc8 = nn.Conv2d(1024, self.n_classes, 1, bias=False)
        self.fc8_sub = nn.Conv2d(1024, self.sub_classes, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.from_scratch_layers = [self.fc8]

        return

    def feature_extractor(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        conv4 = x
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        conv5 = x

        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        x = F.relu(self.fc7(x))
        conv5fc = x

        return conv5fc

    def forward(self, x):

        x = self.feature_extractor(x)
        feat = self.drop7(x)

        x = self.fc8(feat)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, self.n_classes)

        x_sub = self.fc8_sub(feat)
        x_sub = F.avg_pool2d(x_sub, kernel_size=(x_sub.size(2), x_sub.size(3)), padding=0)
        x_sub = x_sub.view(-1, self.sub_classes)


        return x,x_sub


    def cam_output(self, x):

        x = self.feature_extractor(x)
        x = self.fc8_sub(x) ## fc8 was set with bias=False and this is why applying the fc8 is like array multiplication as intructed in equation (1) of paper (1)
        # x = self.fc8(x) ## fc8 was set with bias=False and this is why applying the fc8 is like array multiplication as intructed in equation (1) of paper (1)

        x = F.relu(x)
        x = torch.sqrt(x) ## smoothed by square rooting to obtain a more uniform cam visualization
        return x

    def freeze_layers(self, frozen_stages):

        for layer in self.named_parameters():
            if "conv" in layer[0] and np.int(layer[0].split("conv")[-1][0]) in frozen_stages:
                layer[1].requires_grad = False

    def load_pretrained(self, pth_file):

        weights_dict = torch.load(pth_file)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if
                           k in model_dict and weights_dict[k].shape == model_dict[k].shape}
        #
        # no_pretrained_dict = {k: v for k, v in model_dict.items() if
        #                    not (k in weights_dict) or weights_dict[k].shape != model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        return


    def train_epoch(self, dataloader, optimizer, verbose=True):

        train_loss = 0
        train_loss_cls = 0
        train_loss_subcls=0

        train_accuracy_cls = 0
        train_accuracy_subcls = 0

        self.train()

        for index, data in enumerate(dataloader):

            img = data[1]
            label = data[2]
            sub_label = data[3]

            x, x_sub = self(img)
            loss_cls = F.multilabel_soft_margin_loss(x, label)
            loss_sub_cls = F.multilabel_soft_margin_loss(x_sub, sub_label)

            loss = loss_cls + loss_sub_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss_cls += loss_cls
            train_loss_subcls += loss_sub_cls
            train_loss += loss

            ### adding batch loss into the overall loss
            batch_accuracy_cls = 1 - torch.mean((torch.sum(torch.abs(torch.sigmoid(x)>0.5 - label), axis=0) / x.shape[0]))
            batch_accuracy_subcls = 1 - torch.mean((torch.sum(torch.abs(torch.sigmoid(x_sub)>0.5 - sub_label), axis=0) / x_sub.shape[0]))

            train_accuracy_cls += batch_accuracy_cls
            train_accuracy_subcls += batch_accuracy_subcls

            if verbose:
                ### Printing epoch results
                print('Train Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Loss: {:.4f}\n'
                      'Batch ~ Class Loss: {:.4f}\n'
                      'Batch ~ Sub Class Loss: {:.4f}\n'
                      'Batch ~ Class Accuracy: {:.4f}\n'
                      'Batch ~ Sub Class Accuracy: {:.4f}\n'.format(self.epoch+1, self.epochs,
                                                      index + 1, len(dataloader),
                                                      loss.data.cpu().numpy(),
                                                      loss_cls.data.cpu().numpy(),
                                                      loss_sub_cls.data.cpu().numpy(),
                                                      batch_accuracy_cls.data.cpu().numpy(),
                                                      batch_accuracy_subcls.data.cpu().numpy()))


        self.train_history["loss"].append(train_loss / len(dataloader))
        self.train_history["cls_loss"].append(train_loss_cls / len(dataloader))
        self.train_history["subcls_loss"].append(train_loss_subcls / len(dataloader))

        self.train_history["cls_accuracy"].append(train_accuracy_cls / len(dataloader))
        self.train_history["subcls_accuracy"].append(train_accuracy_subcls / len(dataloader))

        return



    def val_epoch(self, dataloader, verbose=True):


           val_loss = 0
           val_loss_cls = 0
           val_loss_subcls = 0

           val_accuracy_cls = 0
           val_accuracy_subcls = 0
           self.eval()

           with torch.no_grad():

               for index, data in enumerate(dataloader):

                   img = data[1]
                   label = data[2]
                   sub_label = torch.from_numpy(torch.repeat_interleave(label,int(self.sub_classes/self.n_classes),1).data.cpu().numpy()).cuda()

                   x,x_sub = self(img)
                   loss_cls = F.multilabel_soft_margin_loss(x, label)
                   loss_sub_cls = F.multilabel_soft_margin_loss(x_sub, sub_label)

                   loss = loss_cls + loss_sub_cls

                   ### adding batch loss into the overall loss
                   val_loss_cls += loss_cls
                   val_loss_subcls += loss_sub_cls
                   val_loss += loss

                   ### adding batch loss into the overall loss
                   batch_accuracy_cls = 1 - torch.mean((torch.sum(torch.abs(torch.sigmoid(x) > 0.5 - label), axis=0) / x.shape[0]))
                   batch_accuracy_subcls = 1 - torch.mean((torch.sum(torch.abs(torch.sigmoid(x_sub) > 0.5 - sub_label), axis=0) / x_sub.shape[0]))

                   val_accuracy_cls += batch_accuracy_cls
                   val_accuracy_subcls += batch_accuracy_subcls

                   if verbose:
                       ### Printing epoch results
                       print('Val Epoch: {}/{}\n'
                             'Step: {}/{}\n'
                             'Batch ~ Loss: {:.4f}\n'
                             'Batch ~ Class Loss: {:.4f}\n'
                             'Batch ~ Sub Class Loss: {:.4f}\n'
                             'Batch ~ Class Accuracy: {:.4f}\n'
                             'Batch ~ Sub Class Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                                           index + 1, len(dataloader),
                                                                           loss.data.cpu().numpy(),
                                                                           loss_cls.data.cpu().numpy(),
                                                                           loss_sub_cls.data.cpu().numpy(),
                                                                           batch_accuracy_cls.data.cpu().numpy(),
                                                                           batch_accuracy_subcls.data.cpu().numpy()))

               self.val_history["loss"].append(val_loss / len(dataloader))
               self.val_history["cls_loss"].append(val_loss_cls / len(dataloader))
               self.val_history["subcls_loss"].append(val_loss_subcls / len(dataloader))

               self.val_history["cls_accuracy"].append(val_accuracy_cls / len(dataloader))
               self.val_history["subcls_accuracy"].append(val_accuracy_subcls / len(dataloader))

           return


    def extract_cams(self, dataloader, cam_folder, low_a, high_a):

        if not os.path.exists(cam_folder):
            os.makedirs(cam_folder)


        with torch.no_grad():
            for index, data in enumerate(dataloader):
                print(str(index)+" / " + str(len(dataloader)))
                #current_path, imgs, label, windows, orginal_shape, original_img
                original_shape = data[4]
                label = data[2]

                imgs = data[1]
                windows = data[3]
                img_original = data[5][0].data.cpu().numpy()

                final_cam = np.zeros([self.n_classes, original_shape[0], original_shape[1]])
                final_cam_unlabeled = np.zeros([self.n_classes, original_shape[0], original_shape[1]])

                for index, img in enumerate(imgs):

                    window = windows[index]


                    x = self.cam_output(img)


                    x = F.upsample(x, [img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)[0]

                    ## removing the crop window
                    x = x[:, window[0]:window[2], window[1]:window[3]]

                    x = F.upsample(x.unsqueeze(0), [original_shape[0].data.cpu().numpy()[0], original_shape[1].data.cpu().numpy()[0]], mode='bilinear', align_corners=False)[0]

                    ## filter out non-existing classes
                    sub_label = torch.from_numpy(torch.repeat_interleave(label, int(self.sub_classes / self.n_classes),
                                                                         1).data.cpu().numpy()).cuda()




                    cam = x.cpu().numpy() * sub_label.clone().view(self.sub_classes, 1, 1).data.cpu().numpy()
                    # cam = x.cpu().numpy() * label.clone().view(self.n_classes, 1, 1).data.cpu().numpy()

                    cam_unlabeled = x.cpu().numpy()

                    n_clusters = int(self.sub_classes/self.n_classes)

                    cam20 = np.zeros([self.n_classes,cam.shape[1],cam.shape[2]])
                    cam20_unlabeled = np.zeros([self.n_classes,cam.shape[1],cam.shape[2]])

                    for i in range(self.n_classes):
                        cam20[i] = np.amax(cam[i * n_clusters:(i+1) * n_clusters],axis=0)
                        cam20_unlabeled[i] = np.amax(cam_unlabeled[i * n_clusters:(i+1) * n_clusters],axis=0)

                    cam = cam20
                    cam_unlabeled = cam20_unlabeled

                    if index % 2 == 1:
                        cam = np.flip(cam, axis=2)
                        cam_unlabeled = np.flip(cam_unlabeled, axis=2)

                    final_cam += cam
                    final_cam_unlabeled += cam_unlabeled

                ## normalizing final_cam
                denom = np.max(final_cam, (1, 2))
                denom_unlabeled = np.max(final_cam_unlabeled, (1, 2))

                ## when class does not exist then divide by one
                denom += 1 - (denom > 0)
                denom_unlabeled += 1 - (denom_unlabeled > 0)

                final_cam /= denom.reshape(self.n_classes, 1, 1)

                ########## savings cams as dict
                final_cam_dict = {}
                final_cam_dict_unlabeled = {}

                for i in range(self.n_classes):
                    final_cam_dict[i] = final_cam[i]
                    final_cam_dict_unlabeled[i] = final_cam_unlabeled[i]

                existing_cams = np.asarray(list(final_cam_dict.values()))
                existing_cams_unlabeled = np.asarray(list(final_cam_dict_unlabeled.values()))

                # bg_score_low = np.power(1 - np.max(existing_cams, axis=0), low_a)
                # bg_score_high = np.power(1 - np.max(existing_cams, axis=0), high_a)
                #
                # bg_score_low_cams = np.concatenate((np.expand_dims(bg_score_low, 0), existing_cams), axis=0)
                # bg_score_high_cams = np.concatenate((np.expand_dims(bg_score_high, 0), existing_cams), axis=0)
                #
                # crf_score_low = crf_inference(img_original, bg_score_low_cams, labels=bg_score_low_cams.shape[0])
                # crf_score_high = crf_inference(img_original, bg_score_high_cams, labels=bg_score_high_cams.shape[0])
                #
                # crf_low_dict = {}
                # crf_high_dict = {}
                #
                # crf_low_dict[0] = crf_score_low[0]
                # crf_high_dict[0] = crf_score_high[0]
                #
                # for i in range(self.n_classes):
                #     crf_low_dict[i+1] = crf_score_low[i+1]
                #     crf_high_dict[i+1] = crf_score_high[i+1]
                #
                # if not os.path.exists(cam_folder+"low"):
                #     os.makedirs(cam_folder+"low")
                #
                # if not os.path.exists(cam_folder+"high"):
                #     os.makedirs(cam_folder+"high")

                if not os.path.exists(cam_folder + "default"):
                    os.makedirs(cam_folder + "default")


                # np.save(cam_folder+"low/"+data[0][0].split("/")[-1][:-4]+".npy", crf_low_dict)
                # np.save(cam_folder+"high/"+data[0][0].split("/")[-1][:-4]+".npy", crf_high_dict)
                np.save(cam_folder+"default/"+data[0][0].split("/")[-1][:-4]+".npy", existing_cams_unlabeled)



    def evaluate_cams(self,dataloader, cam_folder, gt_mask_folder):

        t_hold = 0.2
        c_num = np.zeros(21)
        c_denom = np.zeros(21)
        gt_masks = []
        preds = []


        for index, data in enumerate(dataloader):

            img = data[1]
            label = data[2]

            img_key = data[0][0].split("/")[-1].split(".")[0]

            ## loading generated cam
            # cam_low = np.load(cam_folder +"low/" + img_key + '.npy').item()
            cam_default = np.load(cam_folder +"default/" + img_key + '.npy')
            # cam_high = np.load(cam_folder +"high/" + img_key + '.npy').item()

            ## considering only the labeled cams
            cam_default = cam_default * label.clone().view(self.n_classes, 1, 1).data.cpu().numpy()

            ## normalizing final_cam
            denom = np.max(cam_default, (1, 2))

            ## when class does not exist then divide by one
            denom += 1 - (denom > 0)

            cam_default /= denom.reshape(self.n_classes, 1, 1)


            bg_score = np.expand_dims(np.ones_like(cam_default[0])*t_hold,0)
            pred = np.argmax(np.concatenate((bg_score, cam_default)), 0)



            ## loading ground truth annotated mask
            gt_mask = Image.open(gt_mask_folder + img_key + '.png')
            gt_mask = np.array(gt_mask)

            preds.append(pred)
            gt_masks.append(gt_mask)


        sc = scores(gt_masks, preds, self.n_classes+1)

        return sc

    def extract_sub_category(self,dataloader,sub_folder):

        features={}
        ids = {}

        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)


        for i in range(20):
            features[i] = []
            ids[i] =[]

        with torch.no_grad():
            for index, data in enumerate(dataloader):

                img = data[1]
                key = data[0][0].split("/")[-1].split(".")[0]
                label = data[2].data.cpu().numpy()
                x = self.feature_extractor(img)
                feat = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0).squeeze().data.cpu().numpy()
                feat /= np.linalg.norm(feat)

                id = np.where(label[0])[0]

                for i in id:
                    ids[i].append(key)
                    features[i].append(feat)

            np.save(sub_folder+"features.npy", features)
            np.save(sub_folder+"ids.npy", ids)


    def visualize_graph(self):

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")

        plt.plot(np.arange(len(self.train_history["loss"])), self.train_history["loss"], label="train")
        plt.plot(np.arange(len(self.val_history["loss"])), self.val_history["loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"loss.png")
        plt.close()

        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Class Loss")
        plt.title("Class Loss Graph")

        plt.plot(np.arange(len(self.train_history["cls_loss"])), self.train_history["cls_loss"], label="train")
        plt.plot(np.arange(len(self.val_history["cls_loss"])), self.val_history["cls_loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"cls_loss.png")
        plt.close()

        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Sub Class Loss")
        plt.title("Sub Class Loss Graph")

        plt.plot(np.arange(len(self.train_history["subcls_loss"])), self.train_history["subcls_loss"], label="train")
        plt.plot(np.arange(len(self.val_history["subcls_loss"])), self.val_history["subcls_loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"subcls_loss.png")
        plt.close()



        ## Plotting accyracy
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Class Accuracy")
        plt.title("Class Accuracy Graph")

        plt.plot(np.arange(len(self.train_history["cls_accuracy"])), self.train_history["cls_accuracy"], label="train")
        plt.plot(np.arange(len(self.val_history["cls_accuracy"])), self.val_history["cls_accuracy"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"cls_accuracy.png")
        plt.close()


        ## Plotting accyracy
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Sub Class Accuracy")
        plt.title("Sub Class Accuracy Graph")

        plt.plot(np.arange(len(self.train_history["subcls_accuracy"])), self.train_history["subcls_accuracy"], label="train")
        plt.plot(np.arange(len(self.val_history["subcls_accuracy"])), self.val_history["subcls_accuracy"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"subcls_accuracy.png")
        plt.close()



    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)):

                if m.weight is not None and m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class VGG16Affinity(nn.Module):
    def __init__(self, fc6_dilation=4):
        super(VGG16Affinity, self).__init__()

        self.n_classes = 20

        self.train_history = {"loss": [],
                              "foreground loss": [],
                              "background loss": [],
                              "irrelevant loss": []}

        self.val_history = {"loss": [],
                              "foreground loss": [],
                              "background loss": [],
                              "irrelevant loss": []}

        self.min_val = np.inf


        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, 3, padding=fc6_dilation, dilation=fc6_dilation)
        self.drop6 = nn.Dropout2d() # 0.5 dropout
        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.drop7 = nn.Dropout2d() # 0.5 dropout

        self.f8_3 = nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = nn.Conv2d(512, 128, 1, bias=False)
        self.f8_5 = nn.Conv2d(1024, 256, 1, bias=False)
        self.gn8_3 = nn.modules.normalization.GroupNorm(8, 64)
        self.gn8_4 = nn.modules.normalization.GroupNorm(16, 128)
        self.gn8_5 = nn.modules.normalization.GroupNorm(32, 256)

        self.f9 = torch.nn.Conv2d(448, 448, 1, bias=False)

        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.kaiming_normal_(self.f8_5.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2]
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f8_5, self.f9]

        self.predefined_featuresize = int(448//8)

        from utils import get_indices_of_pairs
        self.ind_from, self.ind_to = get_indices_of_pairs(5, (self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from); self.ind_to = torch.from_numpy(self.ind_to)

        return

    def feature_extractor(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        conv4 = x
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        conv5 = x

        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        x = F.relu(self.fc7(x))
        conv5fc = x

        return dict({'conv4': conv4, 'conv5': conv5, 'conv5fc': conv5fc})



    def forward(self, x, to_dense=False):
        # source: https://github.com/jiwoon-ahn/psa

        d = self.feature_extractor(x)

        f8_3 = F.elu(self.gn8_3(self.f8_3(d['conv4'])))
        f8_4 = F.elu(self.gn8_4(self.f8_4(d['conv5'])))
        f8_5 = F.elu(self.gn8_5(self.f8_5(d['conv5fc'])))

        x = torch.cat([f8_3, f8_4, f8_5], dim=1)
        x = F.elu(self.f9(x))

        if x.size(2) == self.predefined_featuresize and x.size(3) == self.predefined_featuresize:
            ind_from = self.ind_from
            ind_to = self.ind_to
        else:
            ind_from, ind_to = get_indices_of_pairs(5, (x.size(2), x.size(3)))
            ind_from = torch.from_numpy(ind_from); ind_to = torch.from_numpy(ind_to)

        x = x.view(x.size(0), x.size(1), -1)

        ff = torch.index_select(x, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(x, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))

        if to_dense:
            aff = aff.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = x.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_mat = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                      torch.cat([aff, torch.ones([area]), aff])).to_dense().cuda()
            return aff_mat

        else:
            return aff

    def load_pretrained(self, pth_file):

        weights_dict = torch.load(pth_file)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if
                           k in model_dict and weights_dict[k].shape == model_dict[k].shape}
        #
        # no_pretrained_dict = {k: v for k, v in model_dict.items() if
        #                    not (k in weights_dict) or weights_dict[k].shape != model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        return

    def freeze_layers(self, frozen_stages):

        for layer in self.named_parameters():
            if "conv" in layer[0] and np.int(layer[0].split("conv")[-1][0]) in frozen_stages:
                layer[1].requires_grad = False


    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

    def train_epoch(self, dataloader, optimizer, verbose=True):

        self.train()
        overall_loss= 0
        foreground_loss = 0
        background_loss = 0
        neutral_loss = 0

        for index, data in enumerate(dataloader):
            img = data[0]

            bg_label = data[1][0]
            fg_label = data[1][1]
            neg_label = data[1][2]

            aff = self.forward(img)

            bg_count = torch.sum(bg_label) + 1e-5  ## to avoid diving by zero
            fg_count = torch.sum(fg_label) + 1e-5  ##  to avoid diving by zero
            neg_count = torch.sum(neg_label) + 1e-5  ##  to avoid diving by zero

            bg_loss = torch.sum(- bg_label * torch.log(aff + 1e-5)) / bg_count
            fg_loss = torch.sum(- fg_label * torch.log(aff + 1e-5)) / fg_count
            neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff)) / neg_count

            loss = bg_loss / 4 + fg_loss / 4 + neg_loss / 2  ## basically we count irrelevant loss twice as much

            overall_loss += loss
            foreground_loss += fg_loss
            background_loss += bg_loss
            neutral_loss += neg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose:
                ### Printing epoch results
                print('Train Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Overall loss : {:.4f}\n'
                      'Batch ~ Foreground loss : {:.4f}\n'
                      'Batch ~ Background loss : {:.4f}\n'
                      'Batch ~ Irrelevant loss : {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                          index + 1, len(dataloader),
                                                          loss.data.cpu().numpy(),
                                                          fg_loss.data.cpu().numpy(),
                                                          bg_loss.data.cpu().numpy(),
                                                          neg_loss.data.cpu().numpy(),
                                                          ))

        self.train_history["loss"].append(overall_loss / len(dataloader))
        self.train_history["foreground loss"].append(foreground_loss / len(dataloader))
        self.train_history["background loss"].append(background_loss / len(dataloader))
        self.train_history["irrelevant loss"].append(neutral_loss / len(dataloader))

    def val_epoch(self, dataloader, verbose=True):

        self.eval()
        overall_loss = 0
        foreground_loss = 0
        background_loss = 0
        neutral_loss = 0

        with torch.no_grad():

            for index, data in enumerate(dataloader):
                img = data[0]

                bg_label = data[1][0]
                fg_label = data[1][1]
                neg_label = data[1][2]

                aff = self.forward(img)

                bg_count = torch.sum(bg_label) + 1e-5  ## to avoid diving by zero
                fg_count = torch.sum(fg_label) + 1e-5  ##  to avoid diving by zero
                neg_count = torch.sum(neg_label) + 1e-5  ##  to avoid diving by zero

                bg_loss = torch.sum(- bg_label * torch.log(aff + 1e-5)) / bg_count
                fg_loss = torch.sum(- fg_label * torch.log(aff + 1e-5)) / fg_count
                neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff)) / neg_count

                loss = bg_loss / 4 + fg_loss / 4 + neg_loss / 2  ## basically we count irrelevant loss twice as much

                overall_loss += loss
                foreground_loss += fg_loss
                background_loss += bg_loss
                neutral_loss += neg_loss

                if verbose:
                    ### Printing epoch results
                    print('Val Epoch: {}/{}\n'
                          'Step: {}/{}\n'
                          'Batch ~ Overall loss : {:.4f}\n'
                          'Batch ~ Foreground loss : {:.4f}\n'
                          'Batch ~ Background loss : {:.4f}\n'
                          'Batch ~ Irrelevant loss : {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                                      index + 1, len(dataloader),
                                                                      loss.data.cpu().numpy(),
                                                                      fg_loss.data.cpu().numpy(),
                                                                      bg_loss.data.cpu().numpy(),
                                                                      neg_loss.data.cpu().numpy(),
                                                                      ))

            self.val_history["loss"].append(overall_loss / len(dataloader))
            self.val_history["foreground loss"].append(foreground_loss / len(dataloader))
            self.val_history["background loss"].append(background_loss / len(dataloader))
            self.val_history["irrelevant loss"].append(neutral_loss / len(dataloader))


    def visualize_graph(self):

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")

        plt.plot(np.arange(len(self.train_history["loss"])), self.train_history["loss"], label="train")
        plt.plot(np.arange(len(self.val_history["loss"])), self.val_history["loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"overall_loss.png")
        plt.close()

        ## Plotting accyracy
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Foreground loss")

        plt.plot(np.arange(len(self.train_history["foreground loss"])), self.train_history["foreground loss"], label="train")
        plt.plot(np.arange(len(self.val_history["foreground loss"])), self.val_history["foreground loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"foreground_loss.png")
        plt.close()

        ## Plotting accyracy
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Background loss")

        plt.plot(np.arange(len(self.train_history["background loss"])), self.train_history["background loss"], label="train")
        plt.plot(np.arange(len(self.val_history["background loss"])), self.val_history["background loss"], label="val")


        plt.legend()
        plt.savefig(self.session_name+"background_loss.png")
        plt.close()

        ## Plotting accyracy
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Neutral loss")

        plt.plot(np.arange(len(self.train_history["irrelevant loss"])), self.train_history["irrelevant loss"], label="train")
        plt.plot(np.arange(len(self.val_history["irrelevant loss"])), self.val_history["irrelevant loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"neutral_loss.png")
        plt.close()

#
    def apply_affinity_on_cams(self, dataloader, cam_folder, affinity_cam_a, affinity_cam_beta, affinity_cam_iters):

        # source: https://github.com/jiwoon-ahn/psa

        gt_masks = []
        preds = []


        for index, data in enumerate(dataloader):

            print(str(index)+"/"+str(len(dataloader)))

            img_key = data[0][0].split("/")[-1].split(".")[0]
            img = data[1]

            name = data[0][0]
            label = data[2][0]

            orig_shape = img.shape
            padded_size = (int(np.ceil(img.shape[2] / 8) * 8), int(np.ceil(img.shape[3] / 8) * 8))

            p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
            img = F.pad(img, p2d)

            dheight = int(np.ceil(img.shape[2] / 8))
            dwidth = int(np.ceil(img.shape[3] / 8))


            cam_default = np.load(cam_folder +"default/" + img_key + '.npy')
            ## considering only the labeled cams
            cam_default = cam_default * label.clone().view(self.n_classes, 1, 1).data.cpu().numpy()
            ## normalizing final_cam
            denom = np.max(cam_default, (1, 2))
            ## when class does not exist then divide by one
            denom += 1 - (denom > 0)
            cam_default /= denom.reshape(self.n_classes, 1, 1)

            cam = {}

            label_ = label.data.cpu().numpy().tolist()
            for index,l in enumerate(label_):
                if l == 1:
                    cam[index] = cam_default[index,:,:]

            cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
            for k, v in cam.items():
                if label[k] == 1:
                    cam_full_arr[k + 1] = v

            cam_full_arr[0] = (1 - np.max(cam_full_arr[1:], (0), keepdims=False)) ** affinity_cam_a
            cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

            with torch.no_grad():
                aff_mat = torch.pow(self.forward(img.cuda(), True), affinity_cam_beta)

                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                for _ in range(affinity_cam_iters):
                    trans_mat = torch.matmul(trans_mat, trans_mat)

                cam_full_arr = torch.from_numpy(cam_full_arr)
                cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)

                cam_vec = cam_full_arr.view(21, -1)

                cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
                cam_rw = cam_rw.view(1, 21, dheight, dwidth)

                cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)
                _, cam_rw_pred = torch.max(cam_rw, 1)

                res = np.uint8(cam_rw_pred.cpu().data[0])[:orig_shape[2], :orig_shape[3]]


                gt_mask = Image.open("VOCdevkit/VOC2012/SegmentationClass/" + img_key + '.png')
                gt_mask = np.array(gt_mask)

                preds.append(res)
                gt_masks.append(gt_mask)

        sc = scores(gt_masks, preds, self.n_classes + 1)

        return sc
