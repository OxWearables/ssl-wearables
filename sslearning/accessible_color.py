# RGB values
# Yellow and orange might be very similar, so just pick one of them
# Black is safe
acc_black = [0, 0, 0]
acc_orange = [230, 159, 0]
acc_blue = [86, 180, 233]
acc_green = [0, 158, 115]
acc_yellow = [240, 228, 66]

max_val = 255
norm_acc_black = acc_black
norm_acc_orange = [i / max_val for i in acc_orange]
norm_acc_blue = [i / max_val for i in acc_blue]
norm_acc_green = [i / max_val for i in acc_green]
norm_acc_yellow = [i / max_val for i in acc_yellow]

acc_ui_colors = [
    norm_acc_black,
    norm_acc_orange,
    norm_acc_blue,
    norm_acc_green,
    norm_acc_yellow,
]
