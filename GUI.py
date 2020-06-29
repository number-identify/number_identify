import tkinter as tk  # 使用Tkinter前需要先导入
from PIL import Image, ImageTk
from cnocr import CnOcr

import test
# 第1步，实例化object，建立窗口window
import cv2

global cut
global imgs
global txt
global image_file
global canvas
global c_canvas
global position
window = tk.Tk()
position = 110
txt = '0'

# 第2步，给窗口的可视化起名字
window.title('My Window')

#设定窗口的大小(长 * 宽)
window.geometry('600x400')  # 这里的乘是小x




# 在图形界面上创建 500 * 500 大小的画布并放置各种元素
canvas = tk.Canvas(window, height=500, width=250)
c_canvas = tk.Canvas(window, height=500, width=270)
e_k = tk.Label(window, text='输入照片全称：', font=('Arial', 12), width=27, height=2)
e_k.pack()
e = tk.Entry(window, show=None, font=('Arial', 14))  # 显示成明文形式
e.pack()
l = tk.Label(window, text='原图', bg='green', font=('Arial', 12), width=27, height=2)
l.pack(anchor="w")
k = tk.Label(window, text='车牌提取', bg='green', font=('Arial', 12), width=27, height=2)
k.pack(anchor="e")



# 触发函数
def check():
    global cut
    global image_file
    global canvas
    global imgs
    global c_canvas
    global position
    txt = e.get()
    list = test.color_identify(txt)
    img = Image.open(txt)
    img = img.resize((300, 300))

    image_file = ImageTk.PhotoImage(img)  # 图片位置（相对路径，与.py文件同一文件夹下，也可以用绝对路径，需要给定图片具体绝对路径）
    image = canvas.create_image(100, 0, anchor='n', image=image_file)  # 图片锚定点（n图片顶端的中间点位置）放在画布（250,0）坐标处
    canvas.pack(side='left')

    #判断是否有读取到车牌
    if(len(list) < 3):
        list = test.picture_identify(txt)

    # 设置截取图片
    imgs = Image.open('pai.jpg')
    imgs = imgs.resize((250, 100))
    cut = ImageTk.PhotoImage(imgs)
    c_image = c_canvas.create_image(140, 0, anchor='n', image=cut)  # 图片锚定点（n图片顶端的中间点位置）放在画布（250,0）坐标处
    c_text = c_canvas.create_text(140,position,text = list)
    position = position+15
    c_canvas.pack(side='right')

# 定义一个按钮用来移动指定图形的在画布上的位置
b = tk.Button(window, text='检测', command=check).pack(side="bottom")

# 主窗口循环显示
window.mainloop()