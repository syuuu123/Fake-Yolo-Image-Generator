import math
import os
import random
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
import glob
from icecream import ic
import pyautogui
import datetime
import scipy

class Fake_Img_Gen:
    def __init__(self,root):
        self.root = root
        self.root.title('Fake Image Synthesizer')
        self.root.geometry("1300x800")

        '''
        Canvas Background
        '''
        self.canvas_back = tk.Canvas(self.root, bg="black")
        self.canvas_back.place(x=5,y=40)
        self.canvas_back.config(cursor="tcross")

        '''
        Widget for Canvas Background
        '''
        self.live_feed_mode=None

        self.get_background_file_button = tk.Button(self.root,text= 'Get Background Image',command=lambda: self.load_img_from_file('back'))
        self.get_background_file_button.place(x=5,y=5)

        self.camera_button = tk.Button(self.root,text= 'Turn On Webcam',command=lambda :self.start_thread_loop('webcam'))
        self.camera_button.place(x=150,y=5)

        self.screengrab_button = tk.Button(self.root,text= 'Turn On Screen Grab',command=lambda :self.start_thread_loop('screengrab'))
        self.screengrab_button.place(x=270,y=5)

        '''
        Canvas Subject
        '''
        self.canvas_sub = tk.Canvas(self.root, bg="black",width=200 , height=200)
        self.canvas_sub.place(x=1005,y=40)


        '''
        Widget for Canvas Subject
        '''
        self.get_subject_file_button = tk.Button(self.root,text= 'Get Subject Image', command=lambda: self.load_img_from_file('sub'))
        self.get_subject_file_button.place(x=1005,y=5)



        '''
        Events - Binding
        '''
        self.X_n,self.Y_n = 0,0
        self.canvas_back.bind("<Motion>", self.on_mouse_move)
        self.canvas_back.bind("<ButtonPress-1>", self.on_press)

        '''
        Widgets
        x - 1005
        y - spacing 70
        '''
        widget_x = 1010
        self.size_slider = tk.Scale(self.root, from_=-20, to=2,  orient="horizontal", label="Subject Size",sliderlength=20,length=100)
        self.size_slider.set(0)
        self.size_slider.place(x=widget_x,y=250)
        self.size_slider.bind("<ButtonRelease-1>", self.update_subject_img)

        self.random_size_slider = tk.Scale(self.root, from_=0, to=3,  orient="horizontal", label="Randomize",sliderlength=50,length=100)
        self.random_size_slider.set(0)
        self.random_size_slider.place(x=widget_x+100,y=250)

        self.rotation_slider = tk.Scale(self.root, from_=0, to=360,  orient="horizontal", label="Rotation Angle",sliderlength=20,length=100)
        self.rotation_slider.set(0)
        self.rotation_slider.place(x=widget_x,y=320)
        self.rotation_slider.bind("<ButtonRelease-1>", self.update_subject_img)

        self.random_angle_slider = tk.Scale(self.root, from_=0, to=1,  orient="horizontal", label="Randomize",sliderlength=50,length=100)
        self.random_angle_slider.set(0)
        self.random_angle_slider.place(x=widget_x+100,y=320)

        self.brighten_slider = tk.Scale(self.root, from_=1, to=3,  orient="horizontal", label="Brighten",sliderlength=20,length=100)
        self.brighten_slider.set(1)
        self.brighten_slider.place(x=widget_x,y=390)
        self.brighten_slider.bind("<ButtonRelease-1>", self.update_subject_img)

        self.darken_slider = tk.Scale(self.root, from_=-127, to=127,  orient="horizontal", label="Darken",sliderlength=20,length=100)
        self.darken_slider.set(0)
        self.darken_slider.place(x=widget_x,y=460)
        self.darken_slider.bind("<ButtonRelease-1>", self.update_subject_img)


        self.flip_bgr_slider = tk.Scale(self.root, from_=0, to=1,  orient="horizontal", label="Flip BGR",sliderlength=50,length=100)
        self.flip_bgr_slider.set(0)
        self.flip_bgr_slider.place(x=widget_x,y=530)


        self.dir_path=None
        self.set_destination_dir_button = tk.Button(self.root,text= 'Set Output Path', command=self.set_dest_dir)
        self.set_destination_dir_button.place(x=widget_x,y=600)

    def on_mouse_move(self,event):
        canvas_h,canvas_w = self.canvas_back.winfo_height(),self.canvas_back.winfo_width()
        self.Y_n, self.X_n = round(event.y/canvas_h,5),round(event.x/canvas_w,5)
        #print("Mouse position: (Y:%s X:%s)" % (self.Y_n,self.X_n))

    def set_dest_dir(self):
        self.dir_path = tk.filedialog.askdirectory()

    def load_img_from_file(self,canvas_back_or_sub=None):
        if self.live_feed_mode is not None:
            self.live_feed_mode = None

        self.live_feed_mode = None


        filepath = tk.filedialog.askopenfilename()

        img_data = cv2.imread(filepath)
        img_data_ori = img_data.copy()
        img_data = self.resize_to_fit_frame(img_data)

        if canvas_back_or_sub == 'back':
            self.canvas_back_img_playground = img_data.copy()
            self.canvas_back_img_ori = img_data_ori

            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(img_data)
            self.tk_back_img = ImageTk.PhotoImage(image_pil)
            self.canvas_back.create_image(0, 0, anchor=tk.NW, image=self.tk_back_img)
            self.canvas_back.config(width=self.tk_back_img.width() - 2, height=self.tk_back_img.height() - 2)

            # need to reset - because this tied to BGR flipping
            self.live_feed_mode = None

        elif canvas_back_or_sub == 'sub':
            self.size_slider.set(0)
            self.brighten_slider.set(0)
            self.darken_slider.set(0)
            self.canvas_sub_img_playground = img_data.copy()
            self.canvas_sub_img_ori = img_data_ori

            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(img_data)
            self.tk_sub_img = ImageTk.PhotoImage(image_pil)
            self.canvas_sub.create_image(0, 0, anchor=tk.NW, image=self.tk_sub_img)
            self.canvas_sub.config(width=self.tk_sub_img.width() - 2, height=self.tk_sub_img.height() - 2)

    def resize_to_fit_frame(self,input_img):
        h,w = input_img.shape[0], input_img.shape[1]
        max_dim = max([h,w])
        if max_dim >= 1000:
            scale_val = max_dim/1000
            input_img = cv2.resize(input_img , (int(input_img.shape[1]/scale_val),int(input_img.shape[0]/scale_val)))
        return input_img

    def display_img(self,img):
        '''
        Save image and save annotation file
        '''

        cv2.imshow("['S'] to save",self.resize_to_fit_frame(img))
        key = cv2.waitKey(0)
        if key == ord('s'):
            # Save the image
            time_saved_file = datetime.datetime.now().strftime(("%Y_%m_%d[%H%M%S]"))
            filename = f"fake_image_{time_saved_file}"
            #Save image
            cv2.imwrite(f'{self.dir_path}\\{filename}.jpg', img)
            print('Image saved successfully.')
            #Save annotation
            txt_path = f'{self.dir_path}\\{filename}.txt'
            self.converting_to_yolo_annotation(txt_path)

        cv2.destroyAllWindows()

    def update_subject_img(self,event):

        size_val = self.size_slider.get()
        random_val_add = self.random_size_slider.get()
        size_val = random.randint((-1*random_val_add),random_val_add) + size_val

        if size_val < 0:
            size_val = 1/(-1*size_val)
        rotation_val = random.randint(0,360) if self.random_angle_slider.get()==1 else self.rotation_slider.get()
        brighten_val = self.brighten_slider.get()
        darken_val = self.darken_slider.get()

        print(f'Update val {size_val} {rotation_val} {brighten_val} {darken_val}')
        sub_img_display = self.canvas_sub_img_ori.copy()

        '''Image edit pipeline'''
        sub_img_display = scipy.ndimage.rotate(sub_img_display, rotation_val)
        sub_img_display = cv2.resize(sub_img_display,(int(size_val*sub_img_display.shape[1]),int(size_val*sub_img_display.shape[0]))) if size_val != 0 else sub_img_display
        sub_img_display = cv2.convertScaleAbs(sub_img_display, alpha=brighten_val, beta=darken_val)
        self.canvas_sub_img_playground = sub_img_display.copy()

        '''Convert to pil>tkpil'''


        sub_img_display = cv2.cvtColor(sub_img_display, cv2.COLOR_BGR2RGB)
        sub_img_display = Image.fromarray(sub_img_display)
        self.tk_sub_img = ImageTk.PhotoImage(sub_img_display)
        self.canvas_sub.config(width=self.tk_sub_img.width(), height=self.tk_sub_img.height())
        self.canvas_sub.create_image(0, 0, anchor=tk.NW, image=self.tk_sub_img)

    def remove_background(self):
        input_image = self.canvas_sub_img_playground.copy()
        input_image_ori = self.canvas_sub_img_playground.copy()
        input_image =cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
        input_image = cv2.blur(input_image, (3,3))  # dont use blurring coz the outline is pretty clear

        #_, mask = cv2.threshold(input_image, 70, 255, cv2.THRESH_BINARY)
        preset_values = [(11, 2), (15, -2)]
        pixel, val_sub = preset_values[1]
        mask = cv2.adaptiveThreshold(input_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                            cv2.THRESH_BINARY, pixel, val_sub)

        kernel = np.array([2,2])

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        mask = mask.astype(np.uint8)

        extracted_subject = cv2.bitwise_and(input_image_ori, input_image_ori, mask=mask)


        return extracted_subject,mask

    def on_press(self,event):
        self.update_subject_img(event)

        '''paste the subject img to '''
        background_image = self.canvas_back_img_ori.copy()
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        if self.live_feed_mode=='webcam' or self.live_feed_mode=='screengrab':
            print(self.live_feed_mode,'RGB FLIP APPLIED')
            background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

        extracted_subject_image, mask = self.remove_background()

        '''
        Calculate the middle point and get the width of both sides 
        '''
        Yabs,Xabs = int(background_image.shape[0]*self.Y_n),int(background_image.shape[1]*self.X_n)
        half_y,half_x = int(extracted_subject_image.shape[0]/2),int(extracted_subject_image.shape[1]/2)
        ic(half_y,half_x)
        rem_y,rem_x = (extracted_subject_image.shape[0]-half_y),(extracted_subject_image.shape[1]-half_x)
        ic(rem_y,rem_x)


        y0 = ((Yabs-half_y))
        y1 =((Yabs+rem_y))
        x0 = ((Xabs-half_x))
        x1 = ((Xabs+rem_x))

        '''
        Cropping the background
        '''
        # if negative value - change to 0 and capture the difference
        # if value more than background - change to max - and capture difference
        y0,y0_diff = (0,(-1*y0)) if y0<0 else (y0,0)
        y1,y1_diff = (background_image.shape[0],(y1-background_image.shape[0])) if y1>background_image.shape[0] else (y1,0)
        x0,x0_diff = (0,(-1*x0)) if x0<0 else (x0,0)
        x1,x1_diff = (background_image.shape[1],(x1-background_image.shape[1])) if x1>background_image.shape[1] else (x1,0)

        ic(y0,y0_diff)
        ic(y1,y1_diff)
        ic(x0,x0_diff)
        ic(x1,x1_diff)
        '''
        Cropping the subjective
        '''

        sub_h,sub_w = extracted_subject_image.shape[0],extracted_subject_image.shape[1]
        extraction_frame = extracted_subject_image[0+y0_diff:sub_h-y1_diff,0+x0_diff:sub_w-x1_diff]




        '''Select region for pasting in background image'''
        ROI = background_image[y0:y1,x0:x1]

        inv_mask = cv2.bitwise_not(mask)

        inv_mask = inv_mask[0+y0_diff:sub_h-y1_diff,0+x0_diff:sub_w-x1_diff]

        ROI_img = cv2.bitwise_and(ROI, ROI, mask=inv_mask)

        ROI_img = cv2.add(ROI_img,extraction_frame)

        background_image[y0:y1, x0:x1] = ROI_img



        cv_display = background_image.copy()
        background_image = self.resize_to_fit_frame(background_image)

        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        background_image = Image.fromarray(background_image)
        self.tk_back_img = ImageTk.PhotoImage(background_image)
        self.canvas_back.config(width=self.tk_back_img.width(), height=self.tk_back_img.height())
        self.canvas_back.create_image(0, 0, anchor=tk.NW, image=self.tk_back_img)
        self.display_img(cv_display)

    def converting_to_yolo_annotation(self,txt_path):
        '''
        :return: Np array of YOLO coordinates (X center, Y center , Width, Heihg)
        '''
        base_img = self.canvas_back_img_playground.copy()
        sub_img = self.canvas_sub_img_playground.copy()

        base_height, base_width = base_img.shape[0], base_img.shape[1]
        sub_height, sub_width = sub_img.shape[0], sub_img.shape[1]

        y_center = self.Y_n
        x_center = self.X_n

        buffer = 3
        ratio_height = (sub_height + buffer) / base_height
        ratio_width = (sub_width + buffer) / base_width
        array = np.array((x_center, y_center, ratio_width, ratio_height), dtype=np.float32)

        with open(txt_path,'w') as w:
            str_array = ' '.join([str(x) for x in list(array)])
            w.write(f'0 {str_array}\n')

    def start_thread_loop(self,input_type):
        self.live_feed_mode = input_type
        thread_start = threading.Thread(target=self.live_canvas_feed)
        thread_start.start()

    def live_canvas_feed(self):
        print('running live feed')
        if self.live_feed_mode == 'webcam':
            cap = cv2.VideoCapture(0)
        while self.live_feed_mode is not None:
            # Capture a screenshot using pyautogui
            if self.live_feed_mode == 'webcam':
                ret, img_data = cap.read()
                img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
            elif self.live_feed_mode == 'screengrab':
                time.sleep(1)
                img_data = pyautogui.screenshot()
                img_data = np.array(img_data)

            img_data = self.resize_to_fit_frame(img_data)
            self.canvas_back_img_playground = img_data
            self.canvas_back_img_ori = self.canvas_back_img_playground.copy()
            image_pil = Image.fromarray(img_data)
            self.tk_back_img = ImageTk.PhotoImage(image_pil)
            self.canvas_back.create_image(0, 0, anchor=tk.NW, image=self.tk_back_img)
            self.canvas_back.config(width=self.tk_back_img.width() - 2, height=self.tk_back_img.height() - 2)

if __name__ == "__main__":
    root = tk.Tk()
    app = Fake_Img_Gen(root)
    root.mainloop()
