# art-history-visualization

![plots of artworks by mark rothko](/rothko.png)

This repository has code that I use to visualize collections of paintings, and specifically, to track how their colorfulness or complexity changes across time. While these are simplistic visualizations, I've found them very helpful in my work.

## Instructions

1. Make sure you have [numpy](https://numpy.org/), [scipy](https://scipy.org/), [matplotlib](https://matplotlib.org/), [imageio](https://imageio.readthedocs.io/en/stable/) and [opencv](https://pypi.org/project/opencv-python/) installed.

If you haven't worked with numpy before, [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) is probably a better place to start.

OpenCV can sometimes be a nightmare to install, `pip install opencv-python` contains all the functions that are used in this codebase. I've tested it with OpenCV 4.2.0.

2. Get some art images! My demos use images from WikiArt, which can be downloaded using [this script by lucasdavid](https://github.com/lucasdavid/wikiart/), but any art images will do.

The WikiArt images are arranged into a directory structure as follows
```
images/
    artist-name/
        1900/
            123456.jpg
            123457.jpg
            ...
        unknown-year/
            123458.jpg
            123459.jpg
            123460.jpg
```

3. Compute some function of the images to serve as your X and Y values. I have several measures that I've been using in [`image_measures.py`](https://github.com/samgoree/art-history-visualization/blob/main/image_measures.py). Several of these measures are based on the findings of [Machado et al. 2015](https://cdv.dei.uc.pt/wp-content/uploads/2017/11/mrnscc2015.pdf) and [Hasler and Suesstrunk 2003](https://www.researchgate.net/publication/243135534_Measuring_Colourfulness_in_Natural_Images).

I've been using years for the X values, but you could use any of these measures there as well.

Importantly, these qualities are subjective! Most people perceive them similarly, but different people might have different perception. You should choose the metric which best matches your intuition, and should not treat these visualizations as objective measures of art.

4. Call `visual_plot.visual_plot`. See the demos folder for examples.

## Feedback

Please let me know if you make anything neat with this project! I'm also always happy to talk about visual measures for art, subjectivity, aesthetics and their relationship to computer vision. I'm also happy to help if you run into errors or have a slightly unusual use case. Please contact me at `sgoree [at] iu.edu`.

## License

This work is open source under the [GNU General Public License](https://en.wikipedia.org/wiki/GNU_General_Public_License). If you would like to use this code for commercial purposes, please contact Sam Goree at `sgoree [at] iu.edu`.