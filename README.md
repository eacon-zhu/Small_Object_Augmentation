原图片的jpg文件 和 xml标签文件 分开放置
该代码每次只能对一类别的物体进行小目标数据增强，并最终生成xml文件，
如果想要对多个类别物体进行增强，请更改demo.py文件，在其中使用for循环遍历不同的文件，
    注意：crops文件夹中存放着 某一类 小目标

请按以下顺序运行代码

    PS:注意更改 voc2yolo.py 和 yolo2voc.py 中的映射字典


1.使用 voc2yolo.py 转换成需要的 txt 标签文件

    PS: 注意更改其中的 映射标签
    生成的文件保存再 image_txt 文件夹中
    该 image_txt 文件夹用于2中截图小图片
    同时 该py文件 也可单纯的用于生成 txt标签文件，看你的目的是什么
    
2.使用 crop_img.py 从原图中截取下来目标小图片

    根据image_txt中的yolo格式标签文件 从image中截取小图片保存在 crops 文件夹中，
    同时生成 small.txt 文件保存小图片路径
    生成小图标后 建议对小图标进行人工筛选，因为有些图片截取的并不准确
    运行 clean_crops.py 对筛选后的图片重新生成 small.txt 文件
    
3.将背景图放在 background 文件夹中，同时包含 jpg 和 txt标签文件(当天背景图可以没有txt标签文件)
    
    因为我的 背景图里有目标，所以才有一个txt标签文件
    PS: 如果背景图比较少，请运行 file_fast_copy.py 快速生成多个相同的背景文件

4.使用 demo.py 生成粘贴有小图像的jpg图片，以及对应的txt标签

    请修改 cl 的值 以生成正确的标签
    为了使截取下来的图片能够正常 粘贴到 原图中请使用较大尺寸的图片，至少要1920x1080
    因为原图太小，那么截取下来的小图片的尺寸可能大于 背景原图的尺寸，导致粘不上去，    
    图片存放在 JPEGImages 文件夹中， 标签存放在 save_txt 文件夹中
    
5.使用 yolo2voc.py 生成 数据增强后的 xml 文件

    PS: 注意更改其中的 映射标签
    xml标签文件存放在 Annotations 文件夹中

mosaic.py实现数据的mosaic数据增强
image_crop.py实现数据的随即裁剪



