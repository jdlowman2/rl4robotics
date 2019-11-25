    from gym import logger as gymlogger
    from gym.wrappers import Monitor
    gymlogger.set_level(40) #error only

    import math
    import glob
    import io
    import base64
    from IPython.display import HTML

    from pyvirtualdisplay import Display
    display = Display(visible=0, size=(1400, 900))
    display.start()

    def wrap_env(env):
      env = Monitor(env, './video', force=True)
      return env

    def show_video():
      mp4list = glob.glob('video/*.mp4')
      if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        IPython.display.display(HTML(data='''<video alt="test" autoplay
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'))))
      else:
        print("Could not find video")
