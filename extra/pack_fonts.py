# run extra/genfont.sh first
import os
import torch
import lab

font_dir = os.path.join('extra','fonts')
fonts = [name for name in os.listdir(font_dir) if
    os.path.isdir(os.path.join(font_dir, name))]
fonts = sorted(fonts)

chars = [chr(c) for c in range(ord('a'),ord('z')+1)]
images = []
labels = []

for f, font in enumerate(fonts):
    for c, char in enumerate(chars):
        im_file = os.path.join(font_dir, font, char + '.png')
        im = lab.imread(im_file)
        images.append(im)
        labels.append(c)
        print(f"Added {im_file}")

images = torch.cat(images,0)
labels = torch.tensor(labels)
sets = torch.zeros(len(fonts), len(chars), dtype=torch.int32)
sets[-349:,:] = 1
sets = sets.reshape(-1)

imdb = {
    'meta' : {
        'classes' : chars,
        'sets' : ['train', 'val'],
        'fonts' : fonts,
    },
    'images' : images,
    'labels' : labels,
    'sets' : sets,
}
torch.save(imdb, os.path.join('data', 'charsdb.pth'))

