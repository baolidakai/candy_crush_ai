# -*- coding: utf-8 -*-

"""
@project: pyGames
@author: YangYang_y00343969
@file: chimp.py
@ide: PyCharm
@time: 2019/2/16 15:36
"""
"""
This simple example is used for the line-by-line tutorial
that comes with pygame. It is based on a 'popular' web banner.
Note there are comments here, but for the full explanation,
follow along in the tutorial.
"""
import os
import pygame
from pygame.locals import *
from pygame.compat import geterror

if not pygame.font:
    print('Warning, fonts disabled')
if not pygame.mixer:
    print('Warning, sound disabled')
main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(os.path.pardir, 'data')


# functions to create our resources
def load_image(name, colorkey=None):
    fullname = os.path.join(data_dir, name)
    try:
        image = pygame.image.load(r'' + fullname)
    except pygame.error:
        print('Cannot load image:', fullname)
        raise SystemExit(str(geterror()))
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            # 获取某像素的颜色值
            colorkey = image.get_at((0, 0))
        # 设置颜色，可选的flags参数可以是pygame.RLEACCEL，用来在没有加速的时候提供更好的性能
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()


def load_sound(name):
    class NoneSound:
        def play(self): pass

    if not pygame.mixer or not pygame.mixer.get_init():
        return NoneSound()
    fullname = os.path.join(data_dir, name)
    try:
        sound = pygame.mixer.Sound(r'' + fullname)
    except pygame.error:
        print('Cannot load sound: %s' % fullname)
        raise SystemExit(str(geterror()))
    return sound


# classes for our game objects
class Fist(pygame.sprite.Sprite):
    """moves a clenched fist on the screen, following the mouse"""

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)  # call Sprite initializer
        self.image, self.rect = load_image('fist.bmp', -1)
        self.punching = 0

    def update(self):
        """move the fist based on the mouse position"""
        pos = pygame.mouse.get_pos()
        self.rect.midtop = pos
        if self.punching:
            # 移动图像 move_ip()是移动图像本身，move()方法会生成一个新的图像
            self.rect.move_ip(5, 10)

    def punch(self, target):
        """returns true if the fist collides with the target"""
        if not self.punching:
            self.punching = 1
            # inflate 增长或缩小矩形大小
            hitbox = self.rect.inflate(-5, -5)
            # colliderect 测试两个矩形是否重叠
            return hitbox.colliderect(target.rect)

    def unpunch(self):
        """called to pull the fist back"""
        self.punching = 0


class Chimp(pygame.sprite.Sprite):
    """moves a monkey critter across the screen. it can spin the
       monkey when it is punched."""

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)  # call Sprite intializer
        self.image, self.rect = load_image('chimp.bmp', -1)
        screen = pygame.display.get_surface()
        self.area = screen.get_rect()
        self.rect.topleft = 10, 10
        self.move = 1
        self.dizzy = 0

    def update(self):
        """walk or spin, depending on the monkeys state"""
        if self.dizzy:
            self._spin()
        else:
            self._walk()

    def _walk(self):
        """move the monkey across the screen, and turn at the ends"""
        newpos = self.rect.move((self.move, 0))
        if self.rect.left < self.area.left or \
                self.rect.right > self.area.right:
            self.move = -self.move
            newpos = self.rect.move((self.move, 0))
            self.image = pygame.transform.flip(self.image, 1, 0)
        self.rect = newpos

    def _spin(self):
        """spin the monkey image"""
        center = self.rect.center
        self.dizzy = self.dizzy + 12
        if self.dizzy >= 360:
            self.dizzy = 0
            self.image = self.original
        else:
            rotate = pygame.transform.rotate
            self.image = rotate(self.original, self.dizzy)
        self.rect = self.image.get_rect(center=center)

    def punched(self):
        """this will cause the monkey to start spinning"""
        if not self.dizzy:
            self.dizzy = 1
            self.original = self.image


def main():
    """this function is called when the program starts.
       it initializes everything it needs, then runs in
       a loop until the function returns."""
    # Initialize Everything
    pygame.init()
    screen = pygame.display.set_mode((468, 60))
    # 设置界面的标题
    pygame.display.set_caption('Monkey Fever V1.0')
    # 设置鼠标是否可见 True表示可见，False表示不可见
    pygame.mouse.set_visible(False)

    # Create The Backgound
    background = pygame.Surface(screen.get_size())
    # 转换颜色格式以便更快加载，无参数保证可以和显示窗口保持一样的格式
    background = background.convert()
    # 设置背景色
    background.fill((250, 250, 250))

    # Put Text On The Background, Centered
    if pygame.font:
        # 设置字体大小
        font = pygame.font.Font(None, 36)
        money = 0
        # 设置字体的内容、锯齿、和颜色
        text = font.render("Pummel The Chimp, And Win $" + str(money), 1, (10, 10, 10))
        # 设置text的位置 参数可以设置的属性有 x, y ,with, height, centerx, centery
        textpos = text.get_rect(centerx=background.get_width() / 2)
        # 在backgroud的指定位置展示text
        background.blit(text, textpos)

    # Display The Background
    screen.blit(background, (0, 0))
    # 更新显示到屏幕表面
    pygame.display.flip()

    # Prepare Game Objects
    clock = pygame.time.Clock()
    whiff_sound = load_sound('whiff.wav')
    punch_sound = load_sound('punch.wav')
    chimp = Chimp()
    fist = Fist()
    allsprites = pygame.sprite.RenderPlain((fist, chimp))

    # Main Loop
    going = True
    while going:
        clock.tick(60)

        # Handle Input Events
        for event in pygame.event.get():
            if event.type == QUIT:
                going = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                going = False
            elif event.type == MOUSEBUTTONDOWN:
                if fist.punch(chimp):
                    punch_sound.play()  # punch
                    chimp.punched()
                    money += 100
                else:
                    pass
                    whiff_sound.play()  # miss
            elif event.type == MOUSEBUTTONUP:
                fist.unpunch()

        allsprites.update()

        # Draw Everything

        background.fill((250, 250, 250))
        screen.blit(background, (0, 0))
        text = font.render("Pummel The Chimp, And Win $" + str(money), 1, (10, 10, 10))
        background.blit(text, textpos)
        screen.blit(background, (0, 0))
        allsprites.draw(screen)
        pygame.display.flip()

    pygame.quit()


# Game Over


# this calls the 'main' function when this script is executed
if __name__ == '__main__':
    main()
