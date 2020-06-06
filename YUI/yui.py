import os
import tkinter.messagebox
from tkinter.filedialog import askopenfilename
from tkinter import *
from tkinter.ttk import *
import subprocess

import PIL.Image
import PIL.ImageTk
import cv2
from utils import *
# from detect import*
from datetime import datetime
class LoadDisplay(object):
    pausedisplay = 1  # 클래스간 공통변수
    progressbar = 0

    def __init__(self, win, x, y):
        self.win = win
        self.frame = None
        self.frame_count = 0
        self.x = x
        self.y = y
        self.width = 352
        self.height = 288
        self.video_source = ""  # ""D:/DProgram/Desktop/codes/ffmpeg데이터셋만들기/aaa/changingdata250/000016_1h.h264"
        self.move_x = 0
        self.move_y = 0
        self.zoom_x = 1
        self.zoom_y = 1
        self.vid = cv2.VideoCapture(self.video_source)
        self.name = ""

        if not self.vid.isOpened():
            pass  # raise ValueError("Unable", self.video_source)
        else:
            pass  # self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.canvas = tkinter.Canvas(self.win, width=self.width, height=self.height, bg="white")
        self.canvas.place(x=x, y=y)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.l_click)
        self.canvas.bind("<Button-3>", self.r_click)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<space>", self.keypress)
        self.canvas.bind("<ButtonRelease-1>", self.l_click_off)
        self.canvas.bind("<MouseWheel>", self.mousewheel)

        self.delay = 33
        self.update()

        self.r_popup = Menu(window, tearoff=0)
        self.r_popup.add_command(label="x0.5", command=lambda: self.zoom_change(0.5))
        self.r_popup.add_command(label="x1.0", command=lambda: self.zoom_change(1.0))
        self.r_popup.add_command(label="x1.5", command=lambda: self.zoom_change(1.5))
        self.r_popup.add_command(label="x2.0", command=lambda: self.zoom_change(2.0))

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def zoom_change(self, zoom):
        if zoom == 1.0:
            self.move_x = 0
            self.move_y = 0
        self.zoom_x = zoom
        self.zoom_y = zoom
        frame = cv2.resize(self.frame, None, fx=self.zoom_x, fy=self.zoom_y, interpolation=cv2.INTER_LINEAR)

    def find_ext2(self, name='', inv=False):
        if inv == False:
            if self.name == ".264":
                name = "H.264"
            elif self.name == ".h263":
                name = "H.263"
            elif self.name == ".bit":
                name = "IVC"
            elif self.name == ".bmp":
                name = "BMP"
            elif self.name == ".j2k":
                name = "JPEG2000"
            elif self.name == ".jpg":
                name = "JPEG"
            elif self.name == ".m2v":
                name = "MPEG-2"
            elif self.name == ".mp4":
                name = "H.265"
            elif self.name == ".png":
                name = "PNG"
            elif self.name == ".tiff":
                name = "TIFF"
            elif self.name == ".webm":
                name = "VP8"
            else:
                name = "Error"
        else:
            if name == "H.264":
                name = ".264"
            elif name == ".H263":
                name = "h.263"
            elif name == "IVC":
                name = ".bit"
            elif name == "BMP":
                name = ".bmp"
            elif name == "JPEG2000":
                name = ".j2k"
            elif name == "JPEG":
                name = ".jpg"
            elif name == "MPEG-2":
                name = ".m2v"
            elif name == "H.265":
                name = ".mp4"
            elif name == "PNG":
                name = ".png"
            elif name == "TIFF":
                name = ".tiff"
            elif name == "VP8":
                name = ".webm"
            else:
                name = "Error"
        return name


    def changetext(self, text,text2, default=True):
        text.delete(1.0, END)
        "MPEG-2", "H.263", "H.264", "HEVC", "IVC", "VP8", "JPEG", "JPEG2000", "BMP", "PNG", "TIFF"
        name = self.find_ext2()
        text.insert(CURRENT, name)
        text2.insert(CURRENT, name + " file is loaded\n")
############################

    def changedvideo(self, text, ext, str):
        if str == 'e':
            try:
                self.vid = cv2.VideoCapture('encodeded' + ext)
                text.insert(CURRENT, 'Encoded file loaded\n')

                if not self.vid.isOpened():
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    print("@@@@ Sequence Read error TK")
                else:
                    ret, self.frame = self.get_frame()  # 로드시 초기 1프레임 띄우기
                    self.frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
                    self.zoom_x=1; self.zoom_y=1; self.move_x=0; self.move_y=0;
                    self.frame = cv2.resize(self.frame, None, fx=self.zoom_x, fy=self.zoom_y, interpolation=cv2.INTER_LINEAR)
                    LoadDisplay.pausedisplay = 1
                    self.frame_num_p = 0
            except:
                text.insert(CURRENT, 'reading Encoded file failed')
        else:
            try:
                self.vid = cv2.VideoCapture('reconstructed' + ext)
                text.insert(CURRENT, 'Encoded file loaded\n')

                if not self.vid.isOpened():
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    print("@@@@ Sequence Read error TK")
                else:
                    ret, self.frame = self.get_frame()  # 로드시 초기 1프레임 띄우기
                    self.frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
                    self.zoom_x = 1;
                    self.zoom_y = 1;
                    self.move_x = 0;
                    self.move_y = 0;
                    self.frame = cv2.resize(self.frame, None, fx=self.zoom_x, fy=self.zoom_y,
                                            interpolation=cv2.INTER_LINEAR)
                    LoadDisplay.pausedisplay = 1
                    self.frame_num_p = 0
            except:
                text.insert(CURRENT, 'reading Encoded file failed')


###############################
    def changevideo(self, text,text2):
        self.video_source = askopenfilename(initialdir="G:/PycharmProjects/TKinter/",
                                            filetypes=(("All", "*.*"), ("All Files", "*.*")), title="Choose a file.")
        self.vid = cv2.VideoCapture(self.video_source)
        print(int(self.vid.get(5)))
        self.name = os.path.splitext(self.video_source)[1]
        self.changetext(text,text2)
        print(self.vid)

        if not self.vid.isOpened():
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@ Sequence Read error TK")
        else:
            ret, self.frame = self.get_frame()  # 로드시 초기 1프레임 띄우기
            self.frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
            self.zoom_x=1; self.zoom_y=1; self.move_x=0; self.move_y=0;
            self.frame = cv2.resize(self.frame, None, fx=self.zoom_x, fy=self.zoom_y, interpolation=cv2.INTER_LINEAR)
            LoadDisplay.pausedisplay = 1
            self.frame_num_p = 0


    def get_frame(self):
        if self.vid.isOpened():  # self.vid.set(cv2.CV_CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = self.vid.read()
            LoadDisplay.progressbar = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
            if ret:
                return 2, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # success
            else:
                return 3, None  # 시퀀스 끝 빈 프레임
        else:
            return 0, 0  # 초기 init 상태


    def update(self):
        if LoadDisplay.pausedisplay == 1:
            ret = 3  # pause 기능
        else:
            ret, temframe = self.get_frame()  # Get a frame from the video source

        if ret == 2:  # 일반 재생 시
            self.frame = temframe
            temframe = cv2.resize(temframe, None, fx=self.zoom_x, fy=self.zoom_y, interpolation=cv2.INTER_LINEAR)
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(temframe))
            self.canvas.create_image(self.move_x, self.move_y, image=self.photo, anchor=tkinter.NW)

        if ret == 3:  # 영상의 끝일때 마지막 프레임을 재생하도록
            if self.frame is None:
                pass
            else:
                temframe = cv2.resize(self.frame, None, fx=self.zoom_x, fy=self.zoom_y, interpolation=cv2.INTER_LINEAR)
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(temframe))
                self.canvas.create_image(self.move_x, self.move_y, image=self.photo, anchor=tkinter.NW)

        window.after(self.delay, self.update)  # 반복 호출


    def l_click(self, event):
        self.lock_x = -self.move_x + event.x
        self.lock_y = -self.move_y + event.y


    def r_click(self, event):
        try:
            self.r_popup.tk_popup(event.x_root + 30, event.y_root + 10, 0)
        finally:
            self.r_popup.grab_release()
        pass


    def drag(self, event):
        self.move_x = - (self.lock_x - event.x)
        self.move_y = - (self.lock_y - event.y)


    def keypress(self, event):  # canvas 에선 작동안하나봄
        kp = repr(event.char)
        print("pressed", kp)    # repr(event.char))
        if (kp == 'x'):
            print("pressed x", repr(event.char))

    def l_click_off(self, event):
        if self.lock_x == -self.move_x + event.x and self.lock_y == -self.move_y + event.y:
            if LoadDisplay.pausedisplay == 1:
                LoadDisplay.pausedisplay = 0
            else:
                LoadDisplay.pausedisplay = 1


    def mousewheel(self, event):
        if event.delta > 0:
            self.zoom_x = self.zoom_x * 1.25
            self.zoom_y = self.zoom_y * 1.25
        else:
            self.zoom_x = self.zoom_x * 0.75
            self.zoom_y = self.zoom_y * 0.75


    def touch_slide(self, event):
        if self.vid.isOpened():
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, LoadDisplay.progressbar)
            # ret, frame = self.vid.read()
            # LoadDisplay.progressbar = self.vid.get(cv2.CAP_PROP_POS_AVI_RATIO)
        #     if ret:
        #         return 2, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # success
        #     else:
        #         return 3, None  # 시퀀스 끝 빈 프레임
        # else:
        #     return 0, 0  # 초기 init 상태

##################
    def detect(self, text, scenario, index, name):  # 1이 반전, 2가 xor
        print('1. preparing trained model for codec classification...')
        text.insert(CURRENT, '1. preparing trained model for codec classification...\n')
        now = datetime.now()

        print('%d.%02d.%02d %d:%02d:%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second))
        text.insert(CURRENT, '%d.%02d.%02d %d:%02d:%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second))

        print('scenario = ' + str(scenario) + 'index = ' + str(index))
        text.insert(CURRENT, 'scenario = ' + str(scenario) + 'index = ' + str(index) + '\n')

        print('2. preparing randomly encoded bitstream...')
        text.insert(CURRENT, '2. preparing randomly encoded bitstream...\n')

        now = datetime.now()
        print('%d.%02d.%02d %d:%02d:%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second))
        text.insert(CURRENT, '%d.%02d.%02d %d:%02d:%02d\n' % (now.year, now.month, now.day, now.hour, now.minute, now.second))

        # index = randint(0, len(codec_list) - 1)
        ext = codec_list[index]
        video = bitstring.ConstBitStream(filename=name + ext)

        video.tofile(open('original' + ext, 'wb'))
        video = video.read(video.length).bin
        # original = bitstring.BitStream('0b' + video)
        count = factor(len(video))
        # print(len(video), count)
        if scenario == 1:
            video = encode(video, 'inv')
        elif scenario == 2:
            video = xor_fast(video, count)  # 시나리오에 의한 변조
        elif scenario == 3:
            pass
            # dummy 시나리오 변조 넣기
        video = bitstring.BitStream('0b' + video)
        video.tofile(open("encoded" + ext, 'wb'))

        self.changedvideo(text_1_3, ext, 'e')

    def detect_inv(self, text, name):
        print('Decoding process start..')
        text.insert(CURRENT, 'Decoding process start..\n')

        print(name)

        print('1. preparing randomly encoded bitstream...')
        text.insert(CURRENT, '2. preparing randomly encoded bitstream...\n')
        now = datetime.now()

        print('%d.%02d.%02d %d:%02d:%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second))
        text1 = '%d.%02d.%02d %d:%02d:%02d\n' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        text.insert(CURRENT, text1)
        video = bitstring.ConstBitStream(filename=name[0] + name[1])

        video = video.read(video.length).bin
        # original = bitstring.BitStream('0b' + video)
        count = factor(len(video))
        # print(len(video), count)
        video = bitstring.BitStream('0b' + video)
        video.tofile(open('decoded' + name[1], 'wb'))

        # if scenario == 1:
        #     video = encode(video, 'inv')
        # elif scenario == 2:
        #     video = xor_fast(video, count)  # 시나리오에 의한 변조
        # video = bitstring.BitStream('0b' + video)

        print('2. testing what the codec of encoded bitstream is...')  # 코덱분류
        text.insert(CURRENT, '2. testing what the codec of encoded bitstream is...\n')

        now = datetime.now()
        print('%d.%02d.%02d %d:%02d:%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second))
        text2 = '%d.%02d.%02d %d:%02d:%02d\n' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        text.insert(CURRENT, text2)

        frequency_image = codec_decide(video, 'image')
        frequency_video = codec_decide(video, 'video')
        if frequency_video.index(max(frequency_video)) in [1, 2]:   # 비디오를 믿는 경우 = H.263, H.264
            frequency = frequency_video
        else:                                                       # 이외에는 이미지를 믿어도 무방
            frequency = frequency_image

        print(frequency)
        print('3. testing what the scenario of encoded bitstream is...')  # 시나리오분류
        text.insert(CURRENT, '3. testing what the scenario of encoded bitstream is...\n')

        now = datetime.now()
        print('%d.%02d.%02d %d:%02d:%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second))
        text3 = '%d.%02d.%02d %d:%02d:%02d\n' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        text.insert(CURRENT, text3)

        detected_scenario, video = scenario_detect(frequency, video, count)
        print('The scenario for encoding is...')
        text.insert(CURRENT,'The scenario for encoding is...\n')
        print(scenario_list[detected_scenario])
        text.insert(CURRENT, scenario_list[detected_scenario])

        print('4. reconstructing the encoded bitstream...')
        text.insert(CURRENT, '\n4. reconstructing the encoded bitstream...\n')

        now = datetime.now()

        ext = self.find_ext2(scenario_list[detected_scenario], True)

        print('%d.%02d.%02d %d:%02d:%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second))
        text3 = '%d.%02d.%02d %d:%02d:%02d\n' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        text.insert(CURRENT, text3)

        video = bitstring.BitStream('0b' + video)
        self.video_source = bin2hex(video)
        video.tofile(open("reconstructed" + ext, 'wb'))
        self.changedvideo(text_2_3, ext, 'd')
        print('Please compare the reconstructed file with the encoded file!')
        text.insert(CURRENT, 'Please compare the reconstructed file with the encoded file!\n')
        # print(original == video)
        # print('%d.%02d.%02d %d:%02d:%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second))


def set_slider(slidername, *args):
    t = 0
    for i in args:
        if t < i.frame_count: t = i.frame_count

#  slidername Scale(frame1, from_=0, to=200, orient=HORIZONTAL, length=810)


def scenario_act(event):
    if event.widget.current() == 0:
        pass    # 시나리오별 코드s here
    elif event.widget.current() == 1:
        pass    # 시나리오별 코드s here
    elif event.widget.current() == 2:
        pass    # 시나리오별 코드s here
    elif event.widget.current() == 3:
        pass    # 시나리오별 코드s here
    elif event.widget.current() == 4:
        pass    # 시나리오별 코드s here
    elif event.widget.current() == 5:
        pass    # 시나리오별 코드s here
    elif event.widget.current() == 6:
        ## 시나리오7 예시
        seq1 = vid1.video_source
        seq2 = askopenfilename(initialdir="",filetypes=(("All", "*.*"), ("All Files", "*.*")), title="Choose a file.")
        th = subprocess.Popen("python.exe D:/DProgram/Desktop/더미히든시나리오/fakeke_enc_dec.py %s %s" % (seq1, seq2) )
        # , stdin=subprocess.PIPE, stderr=subprocess.PIPE , stdout=subprocess.PIPE

def testF(combo):
    print(combo.current()) # index가져옴

# with Popen(["ifconfig"], stdout=PIPE) as proc:
#     log.write(proc.stdout.read())
window = tkinter.Tk()
window.title('UI test')
window.geometry("900x700+200+200")

notebook = tkinter.ttk.Notebook(window, width=900, height=600)
notebook.pack()

# Tap 1
frame1 = tkinter.Frame(window)
notebook.add(frame1, text="변조")

Origin_labelframe_1 = tkinter.LabelFrame(frame1, text="Origin")
Modified_labelframe_1 = tkinter.LabelFrame(frame1, text="Modified")
States_labelframe_1 = tkinter.LabelFrame(frame1, text="States")

Origin_labelframe_1.pack()
Modified_labelframe_1.pack()
States_labelframe_1.pack()

# Vertical (y) Scroll Bar
yscrollbar = Scrollbar(States_labelframe_1)
yscrollbar.pack(side="right", fill="both")

# text_1_1 = Text(Origin_labelframe_1, width=50, height=20)
# text_1_2 = Text(Modified_labelframe_1, width=50, height=20)
text_1_3 = Text(States_labelframe_1, width=120, height=10, wrap=NONE, yscrollcommand=yscrollbar.set)
#
# text_1_1.insert(tkinter.INSERT, '''Origin''')
# text_1_2.insert(tkinter.INSERT, '''Modified''')
text_1_3.insert(tkinter.INSERT, '''States''')
#
# text_1_1.pack()
# text_1_2.pack()
text_1_3.pack()

# Configure the scrollbars
yscrollbar.config(command=text_1_3.yview)

slider_1 = Scale(frame1, from_=0, to=200, orient=HORIZONTAL, length=810)
slider_1.pack()

# btn_1_2 = tkinter.Button(frame1, text="ㅁ")
# btn_1_3 = tkinter.Button(frame1, text=">>")

# Tap 2
frame2 = tkinter.Frame(window)
notebook.add(frame2, text="복조")

Origin_labelframe_2 = tkinter.LabelFrame(frame2, text="Modified")
Modified_labelframe_2 = tkinter.LabelFrame(frame2, text="Recovered")
States_labelframe_2 = tkinter.LabelFrame(frame2, text="States")

Origin_labelframe_2.pack()
Modified_labelframe_2.pack()
States_labelframe_2.pack()

# Vertical (y) Scroll Bar
yscrollbar = Scrollbar(States_labelframe_2)
yscrollbar.pack(side="right", fill="both")

# text_2_1 = Text(Origin_labelframe_2, width=50, height=20)
# text_2_2 = Text(Modified_labelframe_2, width=50, height=20)
text_2_3 = Text(States_labelframe_2, width=120, height=10, wrap=NONE, yscrollcommand=yscrollbar.set)

# text_2_1.insert(tkinter.INSERT, '''Modified''')
# text_2_2.insert(tkinter.INSERT, '''Recovered''')
text_2_3.insert(tkinter.INSERT, '''States''')

# text_2_1.pack()
# text_2_2.pack()
text_2_3.pack()

# Configure the scrollbars
yscrollbar.config(command=text_2_3.yview)

# combobox
# combo_1_1 = Combobox(frame1)
# combo_1_1['values'] = ("MPEG-2", "H.263", "H.264", "HEVC", "IVC", "VP8", "JPEG", "JPEG2000", "BMP", "PNG", "TIFF")
# combo_1_1.current(0)  # set the selected item


combo_1_2 = Combobox(frame1)
combo_1_2['values'] = ("Scenario-1", "Scenario-2", "Scenario-3", "Scenario-4", "Scenario-5", "Scenario-6", "Scenario-7")
combo_1_2.bind("<<ComboboxSelected>>", scenario_act)
combo_1_2.current(0)  # set the selected item

# combo_1_1.place(x=150, y=0)
combo_1_2.place(x=350, y=0)

slider_2 = Scale(frame2, from_=0, to=200, orient=HORIZONTAL, length=800)
slider_2.pack()

#


# button click event set
# btn_1 = tkinter.Button(window, text='load file & encode', command=lambda: vid1.changevideo(), compound=LEFT)
# btn_2 = tkinter.Button(window, text='distortion', command=lambda: vid2.changevideo(), compound=LEFT)
# btn_3 = tkinter.Button(window, text='load model & classify', command=lambda: vid3.changevideo(), compound=LEFT)
# btn_4 = tkinter.Button(window, text='recover', command=lambda: vid4.changevideo(), compound=LEFT)

text_1_1 = Text(frame1,width = 10,height=1 )
btn_1_1 = tkinter.Button(frame1, text="load file", command=lambda: vid1.changevideo(text_1_1,text_1_3))
btn_1_2 = tkinter.Button(frame1, text="Encode", command=lambda: vid2.detect(text_1_3, combo_1_2.current()+1, codec_list.index(os.path.splitext(vid1.video_source)[1]), os.path.splitext(vid1.video_source)[0]))
# vid2.detect(text_1_3)
# vid2.detect(text_1_3, combo_1_2.current()+1, codec_list.index(os.path.splitext(vid1.video_source)[1]), os.path.splitext(vid1.video_source)[0])
#detect.main(combo_1_2.current(),3,vid1.video_source)
text_2_1 = Text(frame2,width = 10,height=1 )
btn_2_1 = tkinter.Button(frame2, text="load file", command=lambda: vid3.changevideo(text_2_1,text_2_3))
btn_2_2 = tkinter.Button(frame2, text="Decode", command=lambda: vid4.detect_inv(text_2_3, os.path.splitext(vid3.video_source)))


# button position
text_1_1.place(x = 110, y = 5)
btn_1_1.place(x=0, y=0)
btn_1_2.place(x=53, y=0)
text_2_1.place(x = 110, y = 5)
btn_2_1.place(x=0, y=0)
btn_2_2.place(x=53, y=0)
# btn_1_2.place(x=0, y=350)
# btn_2_2.place(x=0, y=350)
# btn_1_3.place(x=30, y=350)
# btn_2_3.place(x=30, y=350)

# windows positions
Origin_labelframe_1.place(x=0, y=30)
Origin_labelframe_2.place(x=0, y=30)
Modified_labelframe_1.place(x=450, y=30)
Modified_labelframe_2.place(x=450, y=30)
States_labelframe_1.place(x=0, y=450)
States_labelframe_2.place(x=0, y=450)

slider_1.place(x=0, y=400)
slider_2.place(x=0, y=400)

vid1 = LoadDisplay(Origin_labelframe_1, 0, 0)
vid2 = LoadDisplay(Modified_labelframe_1, 0, 0)
vid3 = LoadDisplay(Origin_labelframe_2, 0, 0)
vid4 = LoadDisplay(Modified_labelframe_2, 0, 0)


window.mainloop()


