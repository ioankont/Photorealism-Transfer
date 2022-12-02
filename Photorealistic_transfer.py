from Trainer import run_style_transfer

style_path = '/content/drive/MyDrive/ptuxiakh3/housestyle.png'
content_path = '/content/drive/MyDrive/ptuxiakh3/house.png'
style_seg_path = '/content/drive/MyDrive/ptuxiakh3/housestyleseg.png'
content_seg_path = '/content/drive/MyDrive/ptuxiakh3/seghouse.png'

best, best_loss = run_style_transfer(content_path,
                                     style_path,content_seg_path,style_seg_path, num_iterations=3000,content_weight=1e8,style_weight=0.001,tv_weight=1e-2,affine_weight=8e11)
