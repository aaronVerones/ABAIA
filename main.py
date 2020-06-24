import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import cv2
import pickle
import pprint
import helpers
import numpy as np

## Algorithm Parameters
# how many donut-shaped slices to analyze in each image
NUM_SLICES = 7
# how thick the sections of each slice should be (in pixels)
SLICE_SECTION_WIDTH_PX = 25
# tolerance for how different a mean has to be to count as an artifact
TOLERANCE = 0.7

# place images to analyze here (relative paths)
paths = []
debug = helpers.debug

def main():
  for i in range(len(paths)):
    img = helpers.loadImage(paths[i])
    artifacts = run_artifact_detection(img)
    drawFigure(img)
    print(f'image {i} ({paths[i]}) has {len(artifacts)} artifacts')


"""
## inputs ##
img: a dicom image as loaded by pydicom.dcmread. Must have scaled_pixel_array attribute with values scaled according to hounsfield scale

## vocabulary ##
"slice": a donut-shaped section of an image
"section": an approximately-trapezoid-shaped piece of a slice

## returns ##
returns array of artifacts, where an artifact is 
{
  "slice": int,                 # index of slice
  "section": int,               # index of section
  "bounds": {                   # gives the bounds of the affected section
    "innerRadius": float,
    "outerRadius": float,
    "earlyAngle": float,
    "lateAngle": float,
    "centerpoint": (int, int),  # x, y coordinates of the section containing an artifact
  },
  "sliceMean": float,           # mean of voxel values in slice
  "sectionMean": float,         # mean of voxel values in section
  "deviation": float,           # sectionMean - sliceMean
}

## usage ##
image has artifacts if len(run_artifact_detection(img)) > 0
"""
def run_artifact_detection(img):
  # find the center
  cx, cy, r = helpers.findCenterAndRadius(img)
  # sectionData[sliceNo][sectionNo] gives you array of pixel HU values
  sectionData = [] 
  # try to read from cached dump file if already exists
  fileName = f'{path}.sectionData.dump'
  try:
    with open(fileName, 'rb') as dumpFile:
      sectionData = pickle.load(dumpFile)
  except Exception as e:
    helpers.getSectionData(sectionData, img, NUM_SLICES, cx, cy, r, SLICE_SECTION_WIDTH_PX)
    with open(fileName, 'wb') as dumpFile:
      pickle.dump(sectionData, dumpFile)

  ## get the means
  artifacts = []
  for i in range(len(sectionData)):
    sliceData = sectionData[i]
    for j in range(len(sliceData['sections'])):
      section = sliceData['sections'][j]
      section['mean'] = np.average(section['pixels'])
    sliceMean = np.average([ x['mean'] for x in sliceData['sections'] ])
    sliceData['mean'] = sliceMean
    for section in sliceData['sections']:
      sectionMean = section['mean']
      if sectionMean + TOLERANCE < sliceMean:
        # section mean is more than TOLERANCE less than slice mean
        artifacts.append({
          "slice": i,
          "section": j,
          "bounds": {
            "innerRadius": sliceData['bounds']["innerRadius"],
            "outerRadius": sliceData['bounds']["outerRadius"],
            "earlyAngle": section['bounds']["earlyAngle"],
            "lateAngle": section['bounds']["lateAngle"],
            "centerpoint": helpers.getCenterpoint(
              sliceData['bounds']["innerRadius"],
              sliceData['bounds']["outerRadius"],
              section['bounds']["earlyAngle"],
              section['bounds']["lateAngle"],
              cx, cy
            ),
          },
          "sliceMean": sliceMean,
          "sectionMean": sectionMean,
          "deviation": sectionMean - sliceMean
        })
  return artifacts

def drawFigure(img):
  cx, cy, r = helpers.findCenterAndRadius(img)
  # now that I have the artifacts, I need to draw points on the artifacts
  window_center = 1000 + 10
  window_width = 30
  pixels = img.pixel_array
  sliceWidth = r/NUM_SLICES
  scaled_img = cv2.convertScaleAbs(pixels-window_center, alpha=(255.0 / window_width))
  # Create a figure. Equal aspect so circles look circular
  fig,ax = plt.subplots(1)
  ax.set_aspect('equal')
  # drawBoundaries(NUM_SLICES, sliceWidth, SLICE_SECTION_WIDTH_PX, ax, cx, cy, img, r)
  drawArtifacts(artifacts, sliceWidth, ax)
  ax.imshow(scaled_img, cmap='gray', vmin=0, vmax=255)
  plt.show()

def drawArtifacts(artifacts, sliceWidth, ax):
  for a in artifacts:
    circ = Circle(a['bounds']['centerpoint'], sliceWidth/2, fill=False, color='red')
    ax.add_patch(circ)

def drawBoundaries(numSlices, sliceWidth, sectionWidth, ax, cx, cy, img, r):
  for i in range(numSlices):
    innerRadius = sliceWidth * i
    outerRadius = sliceWidth * (i+1)
    circ = Circle((cx, cy), outerRadius, fill=False, color='yellow')
    ax.add_patch(circ)
    numSections = helpers.getNumSections(img, i, r, numSlices, sectionWidth)
    for j in range(numSections):
      # draw a line segment from innerRadius to outerRadius at bearing earlyAngle
      earlyAngle = 360/numSections * j
      point1 = helpers.polarToRectangular(innerRadius, earlyAngle, cx, cy)
      point2 = helpers.polarToRectangular(outerRadius, earlyAngle, cx, cy)
      plt.plot(
        [ point1[0], point2[0] ],
        [ point1[1], point2[1] ],
        color='yellow'
      )

if __name__ == "__main__":
  main()
