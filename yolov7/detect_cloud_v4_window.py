import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy import random
import os
import copy
import pandas as pd
from models.yolo import Model

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadImagestxt
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, iou_filter
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
#from pytorch_quantization import nn as quant_nn
# import quantization.quantize as quantize



def detect(save_img=False):
    errorcheckint = 0
    opt.annotationMakeFile = True
    # if opt.nqat:
    #     print("qat")
    #     #quant_nn.TensorQuantizer.use_fb_fake_quant = True
    #     quantize.initialize()

    checksavefoler = False
    classesnp = np.zeros((500))
    averagetact = np.zeros((2))
    cnt = 0
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    mosaic = opt.mosaic
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()

    device = select_device(opt.device)

    half = device.type != 'cpu'  # half preci sion only supported on CUDA
    # Load model

    # ckpt = torch.load(weights, map_location='cpu')

    # stat_dict = ckpt['model'].float().state_dict()

    # torch.save(stat_dict, 'yolov7_2.pth')

    model = attempt_load(weights, map_location=device)  # load FP32 model

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size


    if trace:
        model = TracedModel(model, device, opt.img_size)
    if half:
        model.half()  # to FP16



    if opt.check:
        #score 값은 0.1에서 0.05 증가 단계로 0.8까지 지정 
        totladata_images = np.zeros(19,dtype=[('infor', int, 14)])
        
        #TP : 사람이 존재하는 이미지에서 사람이 정확하게 검출된 경우
        #FP : 사람이 존재하지 않는 이미지에서 사람으로 잘못 검출된 경우
        #FN : 사람이 존재하는 이미지에서 사람이 검출되어야 할 상황인데, 사람으로 판단하지 못한 경우
        #TN : 사람이 존재하지 않는 이미지에서 사람이 아닌 것으로 정확하게 판단된 경우
        #TP Rate = TP / (TP + FN)
        #FP Rate = FP / (FP + TN)
        # totladata_images[0] = 모든 이미지 수량
        # totladata_images[1] = 사람이 포함된 이미지 수량 
        # totladata_images[2] = 라벨이 없는 이미지 (배경 이미지)
        # totladata_images[3] = 사람 라벨이 없는 이미지
        # totladata_images[4] = 사람 라벨이 존재하고 사람 검출 개수 및 IOU 정상 이미지 수량 (TP)
        # totladata_images[5] = 사람 라벨이 존재하고 사람 검출 개수 정상 IOU 비정상 이미지 수량 (FN)
        # totladata_images[6] = 사람 라벨이 존재하고 사람 검출 개수 비정상 IOU 정상 이미지 수량 (FN)
        # 단, 검출 개수가 많지만 IOU 필터로 거른 후 개수랑 동등할 때는 (TP)
        # totladata_images[7] = 사람 라벨이 존재하고 사람 검출 개수 비정상(검출 수량이 많을 경우) (FN)
        # IOU 필터 거친 후 계상 
        # totladata_images[8] = 사람 라벨이 존재하고 사람 검출 개수 비정상(검출 수량이 적을 경우) (FN)
        # IOU 필터 거친 후 계상 
        # totladata_images[9] = 사람 라벨이 존재하고 사람 미검출 (FN)
        # totladata_images[10] = 사람 라벨이 존재하지 않고 사람 검출 수량 존재 이미지 수량  (FP)
        # totladata_images[11] = 사람 라벨이 존재하고 않고 사람 비검출 이미지 수량  (TN)
        # totladata_images[12] = 라벨이 존재하지 않고 사람 검출 수량 존재 이미지 수량 (FP)
        # totladata_images[13] = 라벨이 존재하지 않고 사람 비검출 수량 존재 이미지 수량 (TN) 

        #totladata_labels = np.zeros((15))
        totladata_labels = np.zeros(19,dtype=[('infor', float, 15)])
        #totladata_labels[0] = 모든 라벨 개수
        #totladata_labels[1] = 사람 라벨 개수
        #totladata_labels[2] = 사람 검출 개수
        #totladata_labels[3] = 정상 사람 라벨 검출 개수
        #totladata_labels[4] = 정상 사람 라벨 SCORE 값 SUM 
        #totladata_labels[5] = 정상 사람 라벨 IOU 값 SUM
        #totladata_labels[6] = IOU 비정상 사람 라벨 검출 개수
        #totladata_labels[7] = IOU 비정상 사람 라벨 SCORE 값 SUM 
        #totladata_labels[8] = IOU 비정상 사람 라벨 IOU 값 SUM
        #totladata_labels[9] = 라벨이 없는 이미지에서 존재하지 않는데 사람 검출 개수 
        #totladata_labels[10] = 라벨이 없는 이미지에서 존재하지 않는데 사람 검출 SCORE 값 SUM 
        #totladata_labels[11] = 라벨이 없는 이미지에서 존재하지 않는데 사람 검출 IOU 값 SUM
        #totladata_labels[12] = 사람 라벨이 존재하지 않는데 사람 검출 개수
        #totladata_labels[13] = 사람 라벨이 존재하지 않는데 사람 검출 SCORE 값 SUM
        #totladata_labels[14] = 사람 라벨이 존재하지 않는데 사람 검출  IOU 값 SUM


        # checklist[0] = 모든 이미지 수량 
        # checklist[1] = 사람이 포함된 이미지 수량 
        # checklist[2] = 라벨이 없는 이미지 
        # checklist[3] = 사람이 없는 이미지 

        # checklist[4] = 라벨이 없는데 사람 검출 라벨 수량 
        # checklist[5] = 사람 라벨이 존재하고 사람 검출 IOU 정상  
        # checklist[6] = 사람 라벨이 존재하고 사람 검출 IOU 비정상 
        # checklist[7] = 사람 라벨은 없는데 사람 검출 

        # checklist[8] = 라벨이 없는데 사람 검출 라벨의 score 값 sum 
        # checklist[9] = 사람 라벨이 존재하고 사람 검출의 IOU 정상  score 값 sum 
        # checklist[10] = 사람 라벨이 존재하고 사람 검출의 IOU 비정상  score 값 sum 
        # checklist[11] = 사람 라벨은 없는데 사람 검출의 score 값 sum 
        # checklist[12] = 라벨이 없는데 사람 검출 라벨의 IOU 값 sum 
        # checklist[13] = 사람 라벨이 존재하고 사람 검출의 IOU 정상  IOU 값 sum 
        # checklist[14] = 사람 라벨이 존재하고 사람 검출의 IOU 비정상  IOU 값 sum 
        # checklist[15] = 사람 라벨은 없는데 사람 검출의 IOU 값 sum 
        # checklist[16] = 라벨이 없고 사람 미검출 이미지 수량 (정상)
        # checklist[17] = 라벨이 없고 사람 검출 이미지 수량 (오검출)
        # checklist[18] = 사람 라벨이 있고 사람 검출 IOU 정상 이미지 수량 (정상) 
        # checklist[19] = 사람 라벨이 있고 사람 검출 IOU 비정상 이미지 수량 (미검출)
        # checklist[20] = 라벨이 없고 사람 검출 
        # checklist[21] = 사람 라벨은 없는데 사람 검출 이미지 수량 (오검출)    
        # checklist[22] = 사람 라벨은 없는데 사람 미검출 이미지 수량 (정상
        # checklist[23] = 라벨 사람 사람 계수 
        # checklist[24] = 사람 검출 계수
        # checklist[25] = 검출 계수가 라벨 계수보다 많을때 IOU 정상 이미지 수량
        # checklist[26] = 검출 계수가 라벨 계수보다 적을때 IOU 정상 이미지 수량
        # checklist[27] = 검출 계수가 라벨 계수보다 많을때  IOU 비정상 이미지 수량 (미검출)
        # checklist[28] = 검출 계수가 라벨 계수보다 적을때  IOU 비정상 이미지 수량 (미검출)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    txt = False

    if str(source).find('.txt') > 0:
        txt = True
        webcam = False
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    elif txt:
        dataset = LoadImagestxt(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    print(device)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run oncel

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    savecnt = 0
    currentcnt = 0
    total_frame_count = 0

    for path, img, im0s, vid_cap in dataset:
        total_frame_count += 1
        if classesnp[1] > 0:
            classesnp[1] = 0
            
        # 비디오 FPS 확인
        if vid_cap:
            video_fps = int(round(vid_cap.get(cv2.CAP_PROP_FPS)))
        else:
            video_fps = 30  # 이미지 시퀀스의 경우 기본값
            
        # 모드에 따라 프레임 처리 여부 결정
        process_frame = False
        
        if opt.capture_mode == 'frames':
            # N프레임마다 처리
            if total_frame_count % opt.capture_interval == 0:
                process_frame = True
        # else:  # 'minutes' 모드
        #     # N분마다 처리
        #     time_elapsed = total_frame_count / video_fps
        #     minutes_elapsed = time_elapsed / 60
        #     if minutes_elapsed > 0 and minutes_elapsed % opt.capture_interval < (1/video_fps):
        #         process_frame = True
        if opt.capture_mode == 'minutes':
            # N분마다 처리하는 로직
            frames_per_interval = int(video_fps * 60 * opt.capture_interval)  # N분에 해당하는 프레임 수
            # 첫 프레임 또는 정확히 N분 간격의 프레임만 처리
            if total_frame_count == 1 or (total_frame_count % frames_per_interval == 0):
                process_frame = True
            else:
                process_frame = False
        
        # 첫 프레임은 항상 처리
        if total_frame_count == 1:
            process_frame = True
            
        # 처리하지 않을 프레임은 건너뛰기
        if not process_frame:
            continue

        totalframe = dataset.nframes

        if opt.check:
            imgcnt = np.zeros(4)
            personlabel = False
            nolabel = False
            nopersonlabel = False

            checklabelpath1 = path.replace("JPEGImages", "labels")
            checklabelpath = checklabelpath1.replace(".jpg", ".txt")
            first = True
            #오검출 및 미검출 이미지 저장 
            if txt:
               source = os.path.dirname(source)
            if checksavefoler == False:
                for dy in range(len(totladata_images)):
                    scorevalue = (dy * 0.05) + 0.1
                    savepath_nolabel = f'{source}\\save\\{opt.name}\\saveimg\\{str(scorevalue)}\\nolabel\\' 
                    savepath_label = f'{source}\\save\\{opt.name}\\saveimg\\{str(scorevalue)}\\label\\'
                    savepath_label_no_person = f'{source}\\save\\{opt.name}\\saveimg\\{str(scorevalue)}\\label_noperson\\'

                    savelabellist = np.zeros(19)
                    #배경 이미지 
                    savepath_nolabel_TP = f'{savepath_nolabel}\\TN\\' #라벨이 존재하지 않고 사람 비검출 수량 존재 이미지 수량
                    savepath_nolabel_FP = f'{savepath_nolabel}\\FP\\' #라벨이 존재하지 않고 사람 검출 수량 존재 이미지 수량
                    #사람 라벨 있음  
                    savepath_label_TP = f'{savepath_label}\\TP\\' #사람 라벨이 존재하고 사람 검출 개수 및 IOU 정상 이미지
                    savepath_label_FN_Miss = f'{savepath_label}\\FN_Miss\\' #사람 라벨이 존재하고 사람 검출 개수 정상 IOU 비정상 이미지
                    savepath_label_FN_NoDetect= f'{savepath_label}\\FN_NoDetect\\' #사람 라벨이 있고 사람 미검출 (미검출)
                    savepath_label_FN_ERROR = f'{savepath_label}\\FN_ERROR\\' #사람 라벨이 존재하고 사람 검출 개수 비정상 IOU 정상 이미지 수량 

                    savepath_label_FN_OVER_COUNT_ERROR = f'{savepath_label}\\FN_OVER_COUNT_ERROR\\' #사람 라벨이 존재하고 사람 검출 개수 비정상(검출 수량이 많을 경우
                    savepath_label_FN_LOW_COUNT_ERROR = f'{savepath_label}\\FN_LOW_COUNT_ERROR\\' #사람 라벨이 존재하고 사람 검출 개수 비정상(검출 수량이 많을 경우
                    #사람 라벨 없음 
                    savepath_label_no_person_TN = f'{savepath_label_no_person}\\TN\\' #사람 라벨이 존재하고 않고 사람 비검출 이미지 수량
                    savepath_label_no_person_FN_ERROR = f'{savepath_label_no_person}\\FN_ERROR\\' #사람 라벨이 존재하지 않고 사람 검출 수량 존재 이미지 


                    if os.path.isdir(savepath_nolabel_TP) == False:
                        os.makedirs(savepath_nolabel_TP)
                    if os.path.isdir(savepath_nolabel_FP) == False:
                        os.makedirs(savepath_nolabel_FP)

                    if os.path.isdir(savepath_label_TP) == False:
                        os.makedirs(savepath_label_TP)
                    if os.path.isdir(savepath_label_FN_Miss) == False:
                        os.makedirs(savepath_label_FN_Miss)
                    if os.path.isdir(savepath_label_FN_NoDetect) == False:
                        os.makedirs(savepath_label_FN_NoDetect)
                    if os.path.isdir(savepath_label_FN_ERROR) == False:
                        os.makedirs(savepath_label_FN_ERROR)
                    if os.path.isdir(savepath_label_FN_OVER_COUNT_ERROR) == False:
                        os.makedirs(savepath_label_FN_OVER_COUNT_ERROR)
                    if os.path.isdir(savepath_label_FN_LOW_COUNT_ERROR) == False:
                        os.makedirs(savepath_label_FN_LOW_COUNT_ERROR)

                    if os.path.isdir(savepath_label_no_person_TN) == False:
                        os.makedirs(savepath_label_no_person_TN)
                    if os.path.isdir(savepath_label_no_person_FN_ERROR) == False:
                        os.makedirs(savepath_label_no_person_FN_ERROR)
                    checksavefoler = True
            personlabelcnt = 0

            checkminsize = False
            
            if(os.path.exists(checklabelpath)):
                imgcnt[0]+=1 #모든 이미지 수량

                labelpersoninfo = [] #사람 라벨 정보 저장  

                file = open(checklabelpath, 'r')

                while True:
                    annotation = file.readline()
                    annotation.rsplit('\x00')       
                    annotation.replace('\n', '')
                    annotation_split = annotation.split()

                    if (0 == len(annotation_split)):
                        if first:
                            nolabel = True
                            break
                        else:
                            break
                    if first:
                        first=False
                    classid = annotation_split[0]

                    if classid == "0":
                        if opt.minsize:
                           withsize = float(annotation_split[3])
                           heightsize = float(annotation_split[3])
                           if withsize < 0.028 or heightsize < 0.083:
                               checkminsize = True
                               continue
                        else:
                            personlabelcnt+=1
                            totladata_labels[1]
                            personlabel = True
                            annotation_join = " ".join(annotation_split)
                            int_list = list(map(float, annotation_join.split()[1:]))
                            labelpersoninfo.append(int_list)

                if personlabel:
                    imgcnt[1]+=1 #사람이 포함된 이미지 수량 증가 
                else:
                    if nolabel:
                        imgcnt[2]+=1 #라벨이 없는 이미지 (배경 이미지)
                    elif checkminsize:
                        nopersonlabel=True
                        imgcnt[3]+=1 #사람 라벨이 없는 이미지
                    else:
                        nopersonlabel=True
                        imgcnt[3]+=1 #사람 라벨이 없는 이미지


        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        #framecnt = 0
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            #opt.annotationMakeFile = True
            if opt.annotationMakeFile:
                p = Path(path)
                sourcefilename = p.stem  # 확장자 제거한 파일명
                
                # 사용자가 지정한 저장 경로가 있으면 사용, 없으면 기존 방식
                if opt.save_path:
                    savapath = opt.save_path
                    # 저장 경로에 JPEGImages와 labels 폴더 생성
                    labelpath = os.path.join(savapath, 'labels')
                    imgpath = os.path.join(savapath, 'JPEGImages')
                    os.makedirs(labelpath, exist_ok=True)
                    os.makedirs(imgpath, exist_ok=True)
                else:
                    savapath = str(p.parent)  # 부모 디렉토리
            # if opt.annotationMakeFile:
            #     name = str(path).split("\\")
            #     cnt = len(name)
            #     name1 = name[cnt-1]
            #     if name1.find(".avi") > 0:
            #         sourcefilename = name1.replace('.avi', '')
            #         savapath = str(path).replace(name1, '')
            #     if name1.find(".mp4") > 0:
            #         sourcefilename = name1.replace('.mp4', '')
            #         savapath = str(path).replace(name1, '')
            #     if name1.find(".jpg") > 0:
            #         sourcefilename = name1.replace('.jpg', '')
            #         savapath = str(path).replace(name1, '')


            saveimg = copy.deepcopy(im0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            bsave = False

            if opt.check:
                #score 값은 0.1에서 0.05 증가 단계로 0.8까지 지정 
                check_imginfo = np.zeros(19, dtype=[('imageStatus', bool, 14)])
                check_labelinfo = np.zeros(19, dtype=[('predictInfor', float, 15)])
                #imageStatus = TF, FP, FN, TN 표기 
                #imageStatus[4] = 사람 라벨이 존재하고 사람 검출  IOU 정상 라벨 존재 (TP)
                #imageStatus[5] = 사람 라벨이 존재하고 사람 검출  IOU 비정상 라벨 존재 (FN)
                #imageStatus[6] = 사람 라벨이 존재하고 사람 검출  비정상 IOU 정상 라벨 존재 (FN)
                #imageStatus[7] = 사람 라벨이 존재하고 사람 검출 개수 비정상(검출 수량이 많을 경우) 라벨 존재 (FN)
                #imageStatus[8] = 사람 라벨이 존재하고 사람 검출 개수 비정상(검출 수량이 적을 경우)라벨 존재 (FN)
                #imageStatus[9] = 사람 라벨이 존재하고 사람 미검출 (FN)
                #imageStatus[10] = 사람 라벨이 존재하지 않고 사람 검출 라벨 존재  (FP)
                #imageStatus[11] = 사람 라벨이 존재하고 않고 사람 비검출  (TN)
                #imageStatus[12] = 라벨이 존재하지 않고 사람 검출 라벨 존재 (FP)
                #imageStatus[13] = 라벨이 존재하지 않고 사람 비검출 (TN) 

                #predictInfor = 각 사람 검출 결과에 대한 정보 저장 
                #predictInfor[3] = 정상 사람 라벨 검출 개수
                #predictInfor[4] = 정상 사람 라벨 SCORE 값
                #predictInfor[5] = 정상 사람 라벨 IOU 값
                #predictInfor[6] = IOU 비정상 사람 라벨 검출 개수
                #predictInfor[7] = IOU 비정상 사람 라벨 SCORE 값
                #predictInfor[8] = IOU 비정상 사람 라벨 IOU
                #predictInfor[9] = 라벨이 없는 이미지에서 존재하지 않는데 사람 검출 개수 
                #predictInfor[10] = 라벨이 없는 이미지에서 존재하지 않는데 사람 검출 SCORE 값 
                #predictInfor[11] = 라벨이 없는 이미지에서 존재하지 않는데 사람 검출 IOU 값
                #predictInfor[12] = 사람 라벨이 존재하지 않는데 사람 검출 개수
                #predictInfor[13] = 사람 라벨이 존재하지 않는데 사람 검출 SCORE 값
                #predictInfor[14] = 사람 라벨이 존재하지 않는데 사람 검출  IOU 값

                if len(labelpersoninfo) ==1:
                    checklabelist = np.zeros(19, dtype=[('check', bool, (1,))])
                else:
                    checklabelistcnt = len(labelpersoninfo)
                    checklabelist = np.zeros(19, dtype=[('check', bool, checklabelistcnt)])
            
            # persondetectcnt = 0
            # annotationcheck = False

            # if opt.annotationMakeFile:
            #     for *xyxy, conf, cls in reversed(det):
            #             for id in opt.classes:
            #                 if cls == id:
            #                     annotationcheck = True
            #                     break
                            
            # if  annotationcheck == False:
            #     continue

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results

                for *xyxy, conf, cls in reversed(det):
                    if cls == 0 or cls == 2 or cls == 5 or cls == 7:
                       classesnp[1] += 1
                       if opt.check:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            if opt.minsize:
                                if xywh[2] < 0.028 or xywh[3] < 0.083:
                                    continue
                                else:
                                    persondetectcnt+=1
                            else:
                                persondetectcnt+=1

                            correct = False
                            idx = int(conf / 0.05) + 1
                            if len(labelpersoninfo) > 0: #사람 라벨이 있을 경우 
                                result, correct,labelchecklst = iou_filter(xywh, labelpersoninfo)
                                checklabelist['check'][idx:0:-1] |= labelchecklst
                            if nolabel:
                                check_imginfo['imageStatus'][idx:0:-1, 12]=True
                                check_labelinfo['predictInfor'][idx:0:-1, 9]+=1
                                check_labelinfo['predictInfor'][idx:0:-1, 10]+=conf.cpu().numpy()
                            elif personlabel: 
                                    if correct:
                                        check_labelinfo['predictInfor'][idx:0:-1, 3]+=1
                                        check_labelinfo['predictInfor'][idx:0:-1, 4]+=conf.cpu().numpy()
                                        check_labelinfo['predictInfor'][idx:0:-1, 5]+=result  
                                    else: 
                                        check_labelinfo['predictInfor'][idx:0:-1, 6]+=1
                                        check_labelinfo['predictInfor'][idx:0:-1, 7]+=conf.cpu().numpy()
                                        check_labelinfo['predictInfor'][idx:0:-1, 8]+=result                                        
                            elif nopersonlabel:
                                 check_labelinfo['predictInfor'][idx:0:-1, 12]+=1
                                 check_labelinfo['predictInfor'][idx:0:-1, 13]+=conf.cpu().numpy()

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    if opt.annotationMakeFile:
                        if opt.save_path:
                            # 이미 위에서 경로 설정됨
                            pass
                        else:
                            labelpath = os.path.join(savapath, sourcefilename, 'labels')
                            imgpath = os.path.join(savapath, sourcefilename, 'JPEGImages')
                            
                            # 폴더 생성 (중간 디렉토리도 함께)
                            os.makedirs(labelpath, exist_ok=True)
                            os.makedirs(imgpath, exist_ok=True)
                            # if os.path.exists(savapath):
                            #     labelpath = f'{savapath}{sourcefilename}\\labels'
                            #     imgpath = f'{savapath}{sourcefilename}\\JPEGImages'
                            #     if not os.path.exists(f'{savapath}{sourcefilename}'):
                            #         os.mkdir(f'{savapath}{sourcefilename}')
                            #         os.mkdir(labelpath)
                            #         os.mkdir(imgpath)
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh)
                        #saveimgpath = imgpath / p.name
                        labelsave_path = os.path.join(labelpath, f'{sourcefilename}_{frame}_{totalframe}_{savecnt}.txt')
                        with open(labelsave_path, 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        imgsave_path = os.path.join(imgpath, f'{sourcefilename}_{frame}_{totalframe}_{savecnt}.jpg')
                        if bsave == False:
                            # 한글 경로 지원을 위한 cv2.imencode() 사용
                            is_success, buffer = cv2.imencode('.jpg', saveimg)
                            if is_success:
                                buffer.tofile(imgsave_path)
                                bsave = True
                    if save_img or view_img or opt.check:  # Add bbox to image
                        persondelete = False
                        if f'{names[int(cls)]}' == "person":
                           persondelete = False
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2, mosaic=mosaic, persondelete=persondelete)

                if bsave:
                    savecnt+=1
                    print(f'\U0001F63F TotalFrame = {totalframe} save cnt = {savecnt}')
            else:
                # if opt.annotationMakeFile and hasattr(opt, 'save_no_detection') and opt.save_no_detection:
                #     if 'labelpath' in locals() and 'imgpath' in locals():
                #         img_filename = f'{sourcefilename}_{frame}_{totalframe}_{savecnt}_sno_detection.jpg'
                #         img_filepath = os.path.join(imgpath, img_filename)
                #         success = cv2.imwrite(img_filepath, saveimg)
                #         if success:
                #             print(f'Saved (no detection): {img_filepath}')
                #             savecnt += 1
                if opt.check:
                    dc = len(check_imginfo) - 1

                    if nolabel:  # No label and no detection, update index 13
                        check_imginfo['imageStatus'][dc:0:-1, 13] = True

                    elif personlabel:  # Person label available, update index 9
                        check_imginfo['imageStatus'][dc:0:-1, 9] = True

                    elif nopersonlabel:  # No person label, update index 11
                        check_imginfo['imageStatus'][dc:0:-1, 11] = True
                        
                if opt.annotationMakeFile and opt.nodetectionsave:
                    if opt.save_path:
                        # 이미 설정된 경로 사용
                        pass
                    else:
                        labelpath = os.path.join(savapath, sourcefilename, 'labels')
                        imgpath = os.path.join(savapath, sourcefilename, 'JPEGImages')
                        os.makedirs(labelpath, exist_ok=True)
                        os.makedirs(imgpath, exist_ok=True)

            # Print time (inference + NMS)
            averagetact[0] = averagetact[0] + 1E3 * (t2 - t1)
            averagetact[1] =averagetact[1] +  1E3 * (t3 - t2)
            cnt+=1
            average1 = averagetact[0] / cnt
            average2 = averagetact[1] / cnt
            if opt.check == False:
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS average inference : %fms,  average NMX : %fms'%(average1, average2))

            if opt.check:
               savecheck = False
               random_value = random.randint(1, 100)
               if random_value > 95:
                   savecheck = True
                   imgresize = cv2.resize(im0, (640,360))
                   filename = path.strip('\n').split('\\')[-1]
               for dx in range(len(check_imginfo)): 
                totladata_labels['infor'][dx][1]+= personlabelcnt
                totladata_labels['infor'][dx][2]+= persondetectcnt

                totladata_images['infor'][dx][0]+=imgcnt[0]
                totladata_images['infor'][dx][1]+=imgcnt[1]
                totladata_images['infor'][dx][2]+=imgcnt[2]
                totladata_images['infor'][dx][3]+=imgcnt[3]

                totladata_labels['infor'][dx][3]+= check_labelinfo['predictInfor'][dx][3]
                totladata_labels['infor'][dx][4]+= check_labelinfo['predictInfor'][dx][4]
                totladata_labels['infor'][dx][5]+= check_labelinfo['predictInfor'][dx][5]

                totladata_labels['infor'][dx][6]+= check_labelinfo['predictInfor'][dx][6]
                totladata_labels['infor'][dx][7]+= check_labelinfo['predictInfor'][dx][7]
                totladata_labels['infor'][dx][8]+= check_labelinfo['predictInfor'][dx][8]
                
                totladata_labels['infor'][dx][9]+= check_labelinfo['predictInfor'][dx][9]
                totladata_labels['infor'][dx][10]+= check_labelinfo['predictInfor'][dx][10]

                totladata_labels['infor'][dx][12]+= check_labelinfo['predictInfor'][dx][12]
                totladata_labels['infor'][dx][13]+= check_labelinfo['predictInfor'][dx][13]
                
                if nolabel:
                   if check_imginfo['imageStatus'][dx][13]:
                      totladata_images['infor'][dx][13]+=1 #라벨이 존재하지 않고 사람 비검출 수량 존재 이미지 수량 (TN)
                      if savecheck and savelabellist[13] < 20 and opt.checksave:
                         savepath = f'{savepath_nolabel_TP}{filename}'
                         is_success, buffer = cv2.imencode('.jpg', imgresize)
                         if is_success:
                            buffer.tofile(imgsave_path)
                         savelabellist[13]+=1

                   if check_imginfo['imageStatus'][dx][12]: #라벨이 존재하지 않고 사람 검출 수량 존재 이미지 수량 (FP)
                      totladata_images['infor'][dx][12]+=1
                      if savecheck and savelabellist[12] < 20 and opt.checksave:
                         savepath = f'{savepath_nolabel_FP}{filename}'
                         is_success, buffer = cv2.imencode('.jpg', imgresize)
                         if is_success:
                            buffer.tofile(imgsave_path)
                         savelabellist[12]+=1

                elif personlabel:
                   if check_imginfo['imageStatus'][dx][9]:
                        totladata_images['infor'][dx][9]+=1
                        if savecheck and savelabellist[9] < 20 and opt.checksave:
                            savepath = f'{savepath_label_FN_NoDetect}{filename}'
                            is_success, buffer = cv2.imencode('.jpg', imgresize)
                            if is_success:
                                buffer.tofile(imgsave_path)
                            savelabellist[9]+=1
                   else:
                       checkcnt = np.count_nonzero(checklabelist['check'][dx])
                       #checkcnt = np.count_nonzero(checklabelist[dx])
                       if len(labelpersoninfo) < checkcnt:
                           totladata_images['infor'][dx][7]+=1 #사람 라벨이 존재하고 사람 검출 개수 비정상(검출 수량이 많을 경우) (FN)
                           if savecheck and savelabellist[7] < 20 and opt.checksave:
                              savepath = f'{savepath_label_FN_NoDetect}{filename}'
                              is_success, buffer = cv2.imencode('.jpg', imgresize)
                              if is_success:
                                 buffer.tofile(imgsave_path)
                              savelabellist[7]+=1
                       if len(labelpersoninfo) == checkcnt:
                          if check_labelinfo['predictInfor'][dx][6] > 0:
                             totladata_images['infor'][dx][5]+=1 #사람 라벨이 존재하고 사람 검출 개수 정상 IOU 비정상 이미지 수량 (FN)
                             if savecheck and savelabellist[5] < 20 and opt.checksave:
                                savepath = f'{savepath_label_FN_NoDetect}{filename}'
                                is_success, buffer = cv2.imencode('.jpg', imgresize)
                                if is_success:
                                    buffer.tofile(imgsave_path)
                                savelabellist[5]+=1
                          else:
                             totladata_images['infor'][dx][4]+=1 #사람 라벨이 존재하고 사람 검출 개수 및 IOU 정상 이미지 수량 (TP)
                             if savecheck and savelabellist[4] < 20 and opt.checksave:
                                savepath = f'{savepath_label_FN_NoDetect}{filename}'
                                is_success, buffer = cv2.imencode('.jpg', imgresize)
                                if is_success:
                                    buffer.tofile(imgsave_path)
                                savelabellist[4]+=1
                       else:
                           totladata_images['infor'][dx][8]+=1 #사람 라벨이 존재하고 사람 검출 개수 비정상(검출 수량이 적을 경우) (FN)
                           if savecheck and savelabellist[8] < 20 and opt.checksave:
                              savepath = f'{savepath_label_FN_LOW_COUNT_ERROR}{filename}'
                              is_success, buffer = cv2.imencode('.jpg', imgresize)
                              if is_success:
                                 buffer.tofile(imgsave_path)
                              savelabellist[8]+=1

                elif nopersonlabel:    
                     if check_labelinfo['predictInfor'][dx][12] > 0: #라벨이 존재하지 않고 사람 검출 수량 존재 이미지 수량 (FP)
                         totladata_images['infor'][dx][12]+=1
                         if savecheck and savelabellist[12] < 20 and opt.checksave:
                            savepath = f'{savepath_label_no_person_TN}{filename}'
                            is_success, buffer = cv2.imencode('.jpg', imgresize)
                            if is_success:
                                 buffer.tofile(imgsave_path)
                            savelabellist[12]+=1
                     else:
                         totladata_images['infor'][dx][13]+=1 #라벨이 존재하지 않고 사람 비검출 수량 존재 이미지 수량 (TN)
                         if savecheck and savelabellist[13] < 20 and opt.checksave:
                            savepath = f'{savepath_label_no_person_FN_ERROR}{filename}'
                            is_success, buffer = cv2.imencode('.jpg', imgresize)
                            if is_success:
                                 buffer.tofile(imgsave_path)
                            savelabellist[13]+=1
            if opt.check: 
                currentcnt+=1
                print('Total img : %d, current img : %d, last img : %d '%(totalframe,currentcnt,totalframe-currentcnt))
            else:
                print('average inference : %fms,  average NMX : %fms'%(average1, average2))     
            # Stream results
            if view_img:
                imgresize = cv2.resize(im0, (1280,720))
                cv2.imshow(str(p), imgresize)
                cv2.waitKey(1)  # 1 millisecond


            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    if opt.check ==False:
                       is_success, buffer = cv2.imencode('.jpg', im0)
                       if is_success:
                          buffer.tofile(imgsave_path)
#                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter) and view_img:
                            vid_writer.release()  # release previous video writer

                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    
                    # if classesnp[1] > 0:
                    #     text = f'person : {int(classesnp[1])}'
                    #     cv2.putText(im0,str(text),(5, 120), cv2.FONT_HERSHEY_SIMPLEX, 2 ,(255,255,255), 3, cv2.LINE_AA)
                    #     cv2.putText(im0,str(text),(5, 120), cv2.FONT_HERSHEY_SIMPLEX, 2 ,(0,0,255), 2, cv2.LINE_AA)
                    vid_writer.write(im0)
                
    if opt.check:
        print('result data save start')

        for dy in range(len(totladata_images)):
            if dy == 18:
                continue
            threshold = round((dy * 0.05) + 0.1, 2)
            totladata_imagespath_dir = f'{source}{opt.name}\\img\\'
            totladata_labelspath_dir = f'{source}{opt.name}\\label\\'
            if os.path.isdir(totladata_imagespath_dir) == False:
                os.makedirs(totladata_imagespath_dir)

            if os.path.isdir(totladata_labelspath_dir) == False:
                os.makedirs(totladata_labelspath_dir)

            totladata_imagespath = f'{totladata_imagespath_dir}{threshold}_result.txt'
            totladata_labelspath = f'{totladata_labelspath_dir}{threshold}_result.txt'

            np.savetxt(totladata_imagespath, totladata_images['infor'][dy+1], fmt='%d', delimiter=',', header='person detection cnt')
            np.savetxt(totladata_labelspath, totladata_labels['infor'][dy+1], fmt='%d', delimiter=',', header='person detection cnt')
        
        file_path = f'{source}{opt.name}\\img\\result_data.csv'
        
        if os.path.isfile(file_path):
           file_path = f'{source}{opt.name}\\img\\result_data-1.csv'
        
        file_path_label = f'{source}{opt.name}\\label\\result_data_label.csv'
        if os.path.isfile(file_path_label):
           file_path_label = f'{source}{opt.name}\\img\\result_data_label-1.csv'

        df = pd.DataFrame(totladata_images['infor']).transpose()
        df_label = pd.DataFrame(totladata_labels['infor']).transpose()

        df.columns = ["0.0", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5","0.55", "0.6","0.65", "0.7","0.75","0.8","0.85","0.9",""]
        df.index = [
            "모든 이미지 수량", 
            "사람이 포함된 이미지 수량", 
            "라벨이 없는 이미지 (배경 이미지) 수량", 
            "사람 라벨이 없는 이미지 수량", 
            "사람 라벨이 존재하고 사람 검출 개수 및 IOU 정상 이미지 수량(TP)",
            "사람 라벨이 존재하고 사람 검출 개수 정상 IOU 비정상 이미지 수량 (FN)", 
            "사람 라벨이 존재하고 사람 검출 개수 비정상 IOU 정상 이미지 수량 (FN)",
            "사람 라벨이 존재하고 사람 검출 개수 비정상(검출 수량이 많을 경우) (FN)", 
            "사람 라벨이 존재하고 사람 검출 개수 비정상(검출 수량이 적을 경우) (FN)",
            "사람 라벨이 존재하고 사람 미검출 (FN)", 
            "사람 라벨이 존재하지 않고 사람 검출 수량 존재 이미지 수량  (FP)", 
            "사람 라벨이 존재하고 않고 사람 비검출 이미지 수량  (TN)", 
            "라벨이 존재하지 않고 사람 검출 수량 존재 이미지 수량 (FP)", 
            "라벨이 존재하지 않고 사람 비검출 수량 존재 이미지 수량 (TN) "]
        
        df_label.columns = ["0.0", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.45", "0.5","0.55", "0.6","0.65", "0.7","0.75","0.8","0.85","0.9",""]
        df_label.index = ["모든 라벨 개수",
                          "사람 라벨 개수",
                          "사람 검출 개수",
                          "정상 사람 라벨 검출 개수",
                          "정상 사람 라벨 SCORE 값 SUM ",
                          "정상 사람 라벨 IOU 값 SUM",
                          "IOU 비정상 사람 라벨 검출 개수",
                          "IOU 비정상 사람 라벨 SCORE 값 SUM ",
                          "IOU 비정상 사람 라벨 IOU 값 SUM",
                          "라벨이 없는 이미지에서 존재하지 않는데 사람 검출 개수 ",
                          "라벨이 없는 이미지에서 존재하지 않는데 사람 검출 SCORE 값 SUM ",
                          "라벨이 없는 이미지에서 존재하지 않는데 사람 검출 IOU 값 SUM",
                          "사람 라벨이 존재하지 않는데 사람 검출 개수",
                          "사람 라벨이 존재하지 않는데 사람 검출 SCORE 값 SUM",
                          "사람 라벨이 존재하지 않는데 사람 검출  IOU 값 SUM"]

        #np.savetxt(file_path, totladata_images['infor'], delimiter=',', fmt=format_specifier,  header=header_description, encoding='cp949')
        df.to_csv(file_path, index=True, sep=',', header=True, encoding='cp949')
        df_label.to_csv(file_path_label, index=True, sep=',', header=True, encoding='cp949')

        print('result data save done')
        # checklist[0] = 모든 이미지 수량 
        # checklist[1] = 사람이 포함된 이미지 수량 
        # checklist[2] = 라벨이 없는 이미지 
        # checklist[3] = 이미지에 사람이 있고, 라벨과 사람 검출 수량이 같을 경우  
        # checklist[4] = 이미지에 사람이 있고, 라벨보다 사람 검출 수량이 적을 경우 
        # checklist[5] = 이미지에 사람이 있고, 라벨보다 사람 검출 수량이 많을 경우 
        # checklist[6] = 사람 라벨이 없는 수량 
        # checklist[7] = 오류 데이터 (라벨이 없거나, 라벨링 데이터가 이상하거나) 
        # checklist[8] = 이미지에 사람이 없고 사람 검출이 존재 할 경우 
        # checklist[9] = 이미지에 사람이 없고 사람 검출이 없는 경우  
        # checklist[10] = 사람 평균 score 
        
    else:
        resultdatasetpath = f'{source}{opt.name}result.txt'
        np.savetxt(resultdatasetpath, classesnp, fmt='%2d', delimiter=',', header='person detection cnt')
    if save_txt:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
    
    # if opt.nqat:
    #     quant_nn.TensorQuantizer.use_fb_fake_quant = False
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='oneik\\best.pt', help='model.pt path(s)')
    #parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    #parser.add_argument('--source', type=str, default='Z:\\etc\cloud_data_v1\\dataset\\JPEGImages\\', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='z:\\01_SafetyEnv\\one\\video\\', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save-path', type=str, default='', help='custom save path for all outputs')  # 추가
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.15, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--annotationMakeFile', action='store_true', help='annotation make file')
    parser.add_argument('--annotationclass', type=int, default=-1, help='annotation make file')
    parser.add_argument('--mosaic', action='store_true', help='annotation make file')
    parser.add_argument('--check', action='store_true', help='check class detection per image')
    parser.add_argument('--check-iou', type=float, default=0.25, help='object confidence threshold')
    # parser.add_argument('--fps', type=int, default=5, help='use only video')
    parser.add_argument('--capture-mode', type=str, default='frames', choices=['frames', 'minutes'],help='capture mode: frames (every N frames) or minutes (every N minutes)')
    parser.add_argument('--capture-interval', type=int, default=5,help='interval for capture: if mode is frames, every N frames; if mode is minutes, every N minutes')
    parser.add_argument('--minsize', action='store_true', help='size filter width 2.3%, height 8.3%')
    parser.add_argument('--checksave', action='store_true', help='size filter width 2.3%, height 8.3%')
    parser.add_argument('--nodetectionsave', action='store_true', help='size filter width 2.3%, height 8.3%')
    parser.add_argument('--nqat', action='store_true', help='qat model')    
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            # opt.nosave = False
            # opt.minsize = False
            # opt.check = True
            detect()

