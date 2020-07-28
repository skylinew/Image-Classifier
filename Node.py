class node:
    space = 0  # stores count of spaces in an image
    plus = 0  # stores count of pluses in an image
    hashtag = 0  # stores count of hashtags in an image
    ratio = 0 # ratio of space to nonspaces

    def __init__(self, image, label):
        self.image = image
        self.label = label

