 TITLE yolov7 svms

set root=C:\Users\S1SECOM\anaconda3

call %root%\Scripts\activate.bat %root%

call conda activate yolov7

call cd D:\code\YOLOV7

call python mode_wf.py --scratch safe-v4.pt --finetuned guyoung.pt --output test0901.pt --alpha 0.9
call python mode_wf.py --scratch safe-v4.pt --finetuned guyoung.pt --output test0802.pt --alpha 0.8
call python mode_wf.py --scratch safe-v4.pt --finetuned guyoung.pt --output test0703.pt --alpha 0.7
call python mode_wf.py --scratch safe-v4.pt --finetuned guyoung.pt --output test0604.pt --alpha 0.6
call python mode_wf.py --scratch safe-v4.pt --finetuned guyoung.pt --output test0505.pt --alpha 0.5
call python mode_wf.py --scratch safe-v4.pt --finetuned guyoung.pt --output test0406.pt --alpha 0.4
call python mode_wf.py --scratch safe-v4.pt --finetuned guyoung.pt --output test0307.pt --alpha 0.3
call python mode_wf.py --scratch safe-v4.pt --finetuned guyoung.pt --output test0208.pt --alpha 0.2
call python mode_wf.py --scratch safe-v4.pt --finetuned guyoung.pt --output test0109.pt --alpha 0.9
pause