from src.ball_inspector import BallInspector

config_path = "/home/yuval-rubin/Projects/tennis_ball/config/tretorn_serie_plus_control.json"

template_path = "/home/yuval-rubin/Projects/tennis_ball/images/template_new_ball.jpg"

image_path = "/home/yuval-rubin/Projects/tennis_ball/images/1.5h.jpg"



inspector = BallInspector(config_path, template_path)
inspector(image_path, visualize_maps=True)


