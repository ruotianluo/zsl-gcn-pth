from torchvision.datasets.folder import *
import os
import torch

def my_make_dataset(dir, class_to_idx, extensions):
    cached_fn = 'cached_'+dir.replace('/','_').strip('_')+'.pth'
    if os.path.isfile(cached_fn):
        print('Load from cached file list: '+cached_fn)
        return torch.load(cached_fn)

    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    torch.save(images, cached_fn)
    return images

class MyDatasetFolder(DatasetFolder):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = my_make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

class MyImageFolder(MyDatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(MyImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        try:
            img = self.loader(path)
        except:
            return None
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path
