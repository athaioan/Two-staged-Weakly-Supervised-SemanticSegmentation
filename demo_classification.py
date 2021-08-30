import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import importlib
from types import SimpleNamespace
from utils import *
from networks import VGG16,VGG16_sub, VGG16_stage3, VGG16Affinity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

### Setting arguments
args = SimpleNamespace(epochs=25,
                       batch_size=6,
                       lr=0.04,
                       weight_decay=5e-4,
                       input_dim=448,
                       pretrained_weights="pretrained_models/vgg16_20M.pth",
                       voc12_img_folder="VOCdevkit/VOC2012/JPEGImages/",
                       voc12_segm_folder="VOCdevkit/VOC2012/SegmentationClass/",
                       train_set="VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",
                       val_set="VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
                       # train_set="VOCdevkit/VOC2012/ImageSets/Segmentation/test_trash.txt",
                       # val_set="VOCdevkit/VOC2012/ImageSets/Segmentation/test_trash.txt",
                       labels_dict="VOCdevkit/VOC2012/cls_labels.npy",
                       frozen_stages=[1, 2], ## freezing the very generic early convolutional layers
                       low_a=4,
                       high_a=32,
                       cam_folder="CAMS/",

                       stage_1=False,
                       stage_2=True,
                       stage_3=False,
                       stage_affinity=True,

                       ## Stage 2
                       subcategory_folder="Subcategory/",
                       epochs_sub=5,
                       batch_size_sub=8,
                       lr_sub=0.004,
                       weight_decay_sub=5e-4,
                       frozen_stages_sub=[1, 2],  ## freezing the very generic early convolutional layers
                       pretrained_weights_sub="Stage1-Classification/stage_1.pth",
                       n_clusters=2,

                       ## Stage 3
                       epochs_stage3=5,
                       stage3_cams="Stage1-Classification/CAMS/default/",
                       batch_size_stage3=1,
                       lr_stage3=0.005, ## you need to lower it a bit,
                       frozen_stages_stage3 = [],

                       ## stage Affinity
                       low_cams = "Stage1-Classification/CAMS/low/",
                       high_cams = "Stage1-Classification/CAMS/high/",
                       # low_cams = "C:/Users/johny/OneDrive/Deep Learning Project/Paper_1/out_la_crf/",
                       # high_cams = "C:/Users/johny/OneDrive/Deep Learning Project/Paper_1/out_ha_crf/",
                       pretrained_weights_affinity="Stage1-Classification/stage_1.pth",
                       batch_size_affinity=1,
                       lr_affinity=0.01,
                       epochs_affinity=10,
                       stageAffinity_cams="Stage1-Classification/CAMS/",

                       affinity_cam_a=16,
                       affinity_cam_beta=8,
                       affinity_cam_iters=8,


                       )

if args.stage_1:

    # #### Stage 1
    args.session_name = "Stage1-Classification/"
    # #
    # # #### Stage 1.a : Image Classification
    # # ## Constructing the training loader
    # train_loader = VOC2012Dataset(args.train_set, args.labels_dict, args.voc12_img_folder, args.input_dim,
    #                               transform=transforms.Compose([
    #                               transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #                               ]))
    # train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True)
    #
    # ## Constructing the validation loader
    # val_loader = VOC2012Dataset(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    # val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False) ## no point in shufflying the validation data
    #
    # ## Initializing the model
    # model = VGG16(n_classes=20).cuda()
    # model.epochs = args.epochs
    # model.session_name = args.session_name
    # model.load_pretrained(args.pretrained_weights)
    # model.freeze_layers(args.frozen_stages)
    #
    # if not os.path.exists(model.session_name):
    #     os.makedirs(model.session_name)
    #
    #
    # param_groups = model.get_parameter_groups()
    # optimizer = PolyOptimizer([
    #     {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.weight_decay},
    #     {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
    #     {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.weight_decay},
    #     {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
    # ], lr=args.lr, weight_decay=args.weight_decay, max_step=len(train_loader)*args.epochs)
    #
    #
    # for current_epoch in range(model.epochs):
    #
    #     model.epoch = current_epoch
    #
    #     print("Training epoch...")
    #     model.train_epoch(train_loader, optimizer)
    #
    #     print("Validating epoch...")
    #     model.val_epoch(val_loader)
    #     model.visualize_graph()
    #
    #     if model.val_history["loss"][-1] < model.min_val:
    #         print("Saving model...")
    #         model.min_val = model.val_history["loss"][-1]
    #
    #         torch.save(model.state_dict(), model.session_name+"stage_1.pth")
    #
    #
    # ### Stage 1.b : Cam Extraction
    # ## Initializing the model
    model = VGG16(n_classes=20).cuda()
    model.session_name = args.session_name
    model.load_pretrained(model.session_name+"stage_1.pth")
    model.eval()
    #
    # ### Cam extraction
    ## Constructing the cam train loader
    # train_loader_cam = VOC2012DatasetCAM(args.train_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    # train_loader_cam = DataLoader(train_loader_cam, batch_size=1, shuffle=False) ## no point in shufflying
    # print("Extracting CAMS for train...")
    # model.extract_cams(train_loader_cam, model.session_name+args.cam_folder, args.low_a, args.high_a)

    # ## Constructing the cam val evaluation loader
    val_loader_cam = VOC2012DatasetCAM(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    val_loader_cam = DataLoader(val_loader_cam, batch_size=1, shuffle=False) ## no point in shufflying
    print("Extracting CAMS for val...")
    model.extract_cams(val_loader_cam, model.session_name+args.cam_folder, args.low_a, args.high_a)


    ## Cam evaluation
    # Constructing the cam train evaluate loader
    train_loader_eval_cam = VOC2012Dataset(args.train_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    train_loader_eval_cam = DataLoader(train_loader_eval_cam, batch_size=1, shuffle=False) ## no point in shufflying the training data during evaluation

    print("Evaluating CAMS for train...")
    segm_train_accuracy = model.evaluate_cams(train_loader_eval_cam, model.session_name+args.cam_folder, args.voc12_segm_folder)

    print(segm_train_accuracy)


    ## Constructing the cam val evaluate loader
    val_loader_eval_cam = VOC2012Dataset(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    val_loader_eval_cam = DataLoader(val_loader_eval_cam, batch_size=1, shuffle=False) ## no point in shufflying the validation data

    print("Evaluating CAMS for eval...")
    segm_val_accuracy = model.evaluate_cams(val_loader_eval_cam, model.session_name+args.cam_folder, args.voc12_segm_folder)

    print(segm_val_accuracy)

if args.stage_2:

    # #### Stage 2
    args.session_name = "Stage2-Classification/"
    #
    # ### Step 2.a: feature extraction
    model = VGG16(n_classes=20).cuda()
    model.session_name = args.session_name
    model.load_pretrained("Stage1-Classification/"+"stage_1.pth")
    model.eval()

    ## Constructing the training loader for subcategory
    # train_loader_subcategory = VOC2012Dataset(args.train_set, args.labels_dict, args.voc12_img_folder, args.input_dim,
    #                               transform=transforms.Compose([
    #                               transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #                               ]))
    # train_loader_subcategory = DataLoader(train_loader_subcategory, batch_size=1, shuffle=False)
    #
    # print("Extracting Features...")
    # model.extract_sub_category(train_loader_subcategory, model.session_name+args.subcategory_folder)

    # ### Step 2.b: generating subcategories

    features = np.load(model.session_name+args.subcategory_folder+"features.npy").item()
    ids = np.load(model.session_name+args.subcategory_folder+"ids.npy").item()

    # constructing sub-categories
    overall_subcategory_labels = {}
    for i in range(model.n_classes):
        for key in ids[i]:
            overall_subcategory_labels[key] = np.zeros(args.n_clusters*model.n_classes)

    for i in range(model.n_classes):
        current_features = features[i]
        current_ids = ids[i]

        current_features = np.asarray(current_features)
        pca = PCA(n_components=3)
        current_features = pca.fit(current_features.T).components_.T

        k_center = KMeans(n_clusters=args.n_clusters, random_state=0, max_iter=300).fit(current_features)
        sub_labels = args.n_clusters*i + k_center.labels_

        for idx, img_key in enumerate(ids[i]):
            sub_label = sub_labels[idx]
            overall_subcategory_labels[img_key][sub_label] = 1

    np.save(model.session_name+args.subcategory_folder+"cls_sublabels.npy", overall_subcategory_labels)

    ## Step 2.c: Image Classification with subcategories
    # Initializing the model
    args.session_name = "Stage2-Classification/"
    # #
    model = VGG16_sub(n_classes=20, n_subclasses=20*args.n_clusters).cuda()
    model.epochs = args.epochs_sub
    model.session_name = args.session_name
    model.load_pretrained(args.pretrained_weights_sub)
    model.freeze_layers(args.frozen_stages_sub)

    ## Constructing the training subcategory loader
    train_loader_subcategory = VOC2012Dataset_subcategory(args.train_set, args.labels_dict, model.session_name+args.subcategory_folder+"cls_sublabels.npy", args.voc12_img_folder, args.input_dim,
                                  transform=transforms.Compose([
                                  transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                                  ]))
    train_loader_subcategory = DataLoader(train_loader_subcategory, batch_size=args.batch_size_sub, shuffle=True)

    ## Constructing the validation loader
    val_loader_subcategory = VOC2012Dataset(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    val_loader_subcategory = DataLoader(val_loader_subcategory, batch_size=args.batch_size_sub, shuffle=False) ## no point in shufflying the validation data


    param_groups = model.get_parameter_groups()
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr_sub, 'weight_decay': args.weight_decay_sub},
        {'params': param_groups[1], 'lr': 2 * args.lr_sub, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr_sub, 'weight_decay': args.weight_decay_sub},
        {'params': param_groups[3], 'lr': 20 * args.lr_sub, 'weight_decay': 0}
    ], lr=args.lr_sub, weight_decay=args.weight_decay, max_step=len(train_loader_subcategory)*args.epochs)

    for current_epoch in range(model.epochs):

        model.epoch = current_epoch

        print("Training epoch...")
        model.train_epoch(train_loader_subcategory, optimizer)

        print("Validating epoch...")
        model.val_epoch(val_loader_subcategory)
        model.visualize_graph()

        if model.val_history["loss"][-1] < model.min_val:
            print("Saving model...")
            model.min_val = model.val_history["loss"][-1]

            torch.save(model.state_dict(), model.session_name+"stage_2.pth")

    # ### Stage 1.d : Cam Extraction
    # ## Initializing the model
    args.session_name = "Stage2-Classification/"
    model = VGG16_sub(n_classes=20, n_subclasses=20*args.n_clusters).cuda()
    model.session_name = args.session_name
    model.load_pretrained(model.session_name+"stage_2.pth")
    model.eval()
    #
    # ### Cam extraction
    # # ## Constructing the cam train loader
    train_loader_cam = VOC2012DatasetCAM(args.train_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    train_loader_cam = DataLoader(train_loader_cam, batch_size=1, shuffle=False) ## no point in shufflying
    print("Extracting CAMS for train...")
    model.extract_cams(train_loader_cam, model.session_name+args.cam_folder, args.low_a, args.high_a)

    # ## Constructing the cam val evaluation loader
    val_loader_cam = VOC2012DatasetCAM(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    val_loader_cam = DataLoader(val_loader_cam, batch_size=1, shuffle=False) ## no point in shufflying
    print("Extracting CAMS for val...")
    model.extract_cams(val_loader_cam, model.session_name+args.cam_folder, args.low_a, args.high_a)


    # Constructing the cam train evaluate loader
    train_loader_eval_cam = VOC2012Dataset(args.train_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    train_loader_eval_cam = DataLoader(train_loader_eval_cam, batch_size=1, shuffle=False) ## no point in shufflying the training data during evaluation

    print("Evaluating CAMS for train...")
    segm_train_accuracy = model.evaluate_cams(train_loader_eval_cam, model.session_name+args.cam_folder, args.voc12_segm_folder)

    print(segm_train_accuracy)

    # # # ## Constructing the cam val evaluate loader
    val_loader_eval_cam = VOC2012Dataset(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    val_loader_eval_cam = DataLoader(val_loader_eval_cam, batch_size=1, shuffle=False) ## no point in shufflying the validation data

    print("Evaluating CAMS for eval...")
    segm_val_accuracy = model.evaluate_cams(val_loader_eval_cam, model.session_name+args.cam_folder, args.voc12_segm_folder)

    print(segm_val_accuracy)


if args.stage_3:

    # #### Stage 3
    args.session_name = "Stage3-Classification/"
    # #
    # #### Stage 3.a : Image Classification
    # ## Constructing the training loader
    # train_loader = VOC2012Dataset_stage3(args.train_set, args.stage3_cams, args.labels_dict, args.voc12_img_folder, args.input_dim,
    #                               transform=transforms.Compose([
    #                               transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #                               ]))
    # train_loader = DataLoader(train_loader, batch_size=args.batch_size_stage3, shuffle=True)
    #
    # ## Constructing the validation loader
    # val_loader = VOC2012Dataset_stage3(args.val_set, args.stage3_cams, args.labels_dict, args.voc12_img_folder, args.input_dim)
    # val_loader = DataLoader(val_loader, batch_size=args.batch_size_stage3, shuffle=False) ## no point in shufflying the validation data
    #
    # ## Initializing the model
    # model = VGG16_stage3(n_classes=20).cuda()
    # model.epochs = args.epochs_stage3
    # model.session_name = args.session_name
    # model.load_pretrained("Stage1-Classification/"+"stage_1.pth")
    # model.freeze_layers(args.frozen_stages_stage3)
    #
    # if not os.path.exists(model.session_name):
    #     os.makedirs(model.session_name)
    #
    #
    # param_groups = model.get_parameter_groups()
    # optimizer = PolyOptimizer([
    #     {'params': param_groups[0], 'lr': args.lr_stage3, 'weight_decay': args.weight_decay},
    #     {'params': param_groups[1], 'lr': 2 * args.lr_stage3, 'weight_decay': 0},
    #     {'params': param_groups[2], 'lr': 10 * args.lr_stage3, 'weight_decay': args.weight_decay},
    #     {'params': param_groups[3], 'lr': 20 * args.lr_stage3, 'weight_decay': 0}
    # ], lr=args.lr_stage3, weight_decay=args.weight_decay, max_step=len(train_loader)*args.epochs)
    #
    #
    # for current_epoch in range(model.epochs):
    #
    #     model.epoch = current_epoch
    #
    #     print("Training epoch...")
    #     model.train_epoch(train_loader, optimizer)
    #
    #     print("Validating epoch...")
    #     model.val_epoch(val_loader)
    #     model.visualize_graph()
    #
    #     if model.val_history["loss"][-1] < model.min_val:
    #         print("Saving model...")
    #         model.min_val = model.val_history["loss"][-1]
    #
    #         torch.save(model.state_dict(), model.session_name+"stage_3.pth")


  # ### Stage 3.b : Cam Extraction
    # ## Initializing the model
    model = VGG16_stage3(n_classes=20).cuda()
    model.session_name = args.session_name
    model.load_pretrained(model.session_name+"stage_3.pth")
    model.eval()
    #
    # ### Cam extraction
    ## Constructing the cam train loader
    train_loader_cam = VOC2012DatasetCAM_stage3(args.train_set, args.stage3_cams, args.labels_dict, args.voc12_img_folder, args.input_dim)
    train_loader_cam = DataLoader(train_loader_cam, batch_size=1, shuffle=False) ## no point in shufflying
    print("Extracting CAMS for train...")
    model.extract_cams(train_loader_cam, model.session_name+args.cam_folder, args.low_a, args.high_a)

    ## Cam evaluation
    # Constructing the cam train evaluate loader
    train_loader_eval_cam = VOC2012Dataset(args.train_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    train_loader_eval_cam = DataLoader(train_loader_eval_cam, batch_size=1, shuffle=False) ## no point in shufflying the training data during evaluation

    print("Evaluating CAMS for train...")
    segm_train_accuracy = model.evaluate_cams(train_loader_eval_cam, model.session_name+args.cam_folder, args.voc12_segm_folder)

    print(segm_train_accuracy)

    # ## Constructing the cam val evaluation loader
    val_loader_cam = VOC2012DatasetCAM_stage3(args.val_set, args.stage3_cams, args.labels_dict, args.voc12_img_folder, args.input_dim)
    val_loader_cam = DataLoader(val_loader_cam, batch_size=1, shuffle=False) ## no point in shufflying
    print("Extracting CAMS for val...")
    model.extract_cams(val_loader_cam, model.session_name+args.cam_folder, args.low_a, args.high_a)

    ## Constructing the cam val evaluate loader
    val_loader_eval_cam = VOC2012Dataset(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    val_loader_eval_cam = DataLoader(val_loader_eval_cam, batch_size=1, shuffle=False) ## no point in shufflying the validation data

    print("Evaluating CAMS for eval...")
    segm_val_accuracy = model.evaluate_cams(val_loader_eval_cam, model.session_name+args.cam_folder, args.voc12_segm_folder)

    print(segm_val_accuracy)


### Training Affinity: 1.a
if args.stage_affinity:
    args.session_name = "Stage_Affinity-Segmentation/"


    # train_dataset = VOC2012DatasetAffinity(args.train_set, args.voc12_img_folder, args.input_dim, label_la_dir=args.low_cams,
    #                                        label_ha_dir=args.high_cams, radius=5, transform=transforms.Compose([
    #                               transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #                               ]))
    #
    # train_loader_affinity = DataLoader(train_dataset, batch_size=args.batch_size_affinity, shuffle=True)
    #
    # val_dataset = VOC2012DatasetAffinity(args.train_set, args.voc12_img_folder, args.input_dim, label_la_dir=args.low_cams,
    #                                        label_ha_dir=args.high_cams, radius=5)
    #
    # val_loader_affinity = DataLoader(val_dataset, batch_size=args.batch_size_affinity, shuffle=False)
    #
    #
    # model = VGG16Affinity(4).cuda()
    # model.session_name = args.session_name
    # model.load_pretrained(args.pretrained_weights_affinity)
    # model.epochs = args.epochs_affinity
    #
    # if not os.path.exists(model.session_name):
    #     os.makedirs(model.session_name)
    #
    # param_groups = model.get_parameter_groups()
    # optimizer = PolyOptimizer([
    #     {'params': param_groups[0], 'lr': args.lr_affinity, 'weight_decay': args.weight_decay},
    #     {'params': param_groups[1], 'lr': 2 * args.lr_affinity, 'weight_decay': 0},
    #     {'params': param_groups[2], 'lr': 10 * args.lr_affinity, 'weight_decay': args.weight_decay},
    #     {'params': param_groups[3], 'lr': 20 * args.lr_affinity, 'weight_decay': 0}
    # ], lr=args.lr_affinity, weight_decay=args.weight_decay, max_step=len(train_loader_affinity)*args.epochs)
    #
    #
    # for epoch in range(model.epochs):
    #
    #     model.epoch = epoch
    #     print("Training epoch...")
    #     model.train_epoch(train_loader_affinity, optimizer)
    #
    #     print("Validating epoch...")
    #     model.val_epoch(val_loader_affinity)
    #     model.visualize_graph()
    #
    #     if model.val_history["loss"][-1] < model.min_val:
    #         print("Saving model...")
    #         model.min_val = model.val_history["loss"][-1]
    #
    #         torch.save(model.state_dict(), model.session_name+"stage_Affinity.pth")

    # ### Applying Affinity
    model = VGG16Affinity(4).cuda()
    model.session_name = args.session_name
    model.load_pretrained(args.session_name+"stage_Affinity.pth")

    model.eval()

    train_loader_eval_cam = VOC2012Dataset_cam_affinity(args.train_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    train_loader_cam = DataLoader(train_loader_eval_cam, batch_size=1, shuffle=False) ## no point in shufflying the training data during evaluation

    val_loader_eval_cam = VOC2012Dataset_cam_affinity(args.val_set, args.labels_dict, args.voc12_img_folder, args.input_dim)
    val_loader_cam = DataLoader(val_loader_eval_cam, batch_size=1, shuffle=False) ## no point in shufflying the training data during evaluation



    sc = model.apply_affinity_on_cams(train_loader_cam, args.stageAffinity_cams,args.affinity_cam_a, args.affinity_cam_beta, args.affinity_cam_iters)

    print(sc)
