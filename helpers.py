import pprint
from scipy.spatial import distance
import random
import pydicom as pd
import numpy as np
from math import floor, pi, isnan, cos, sin

pp = pprint.PrettyPrinter(indent=2)

def noop():
  return None

def debug(
  *values: object, sep = None, end = None, file = None, flush: bool = None,
) -> None:
  # debug(*values, sep=sep, end=end, file=file, flush=flush)
  pass

# loads a dicom image given a path
# also scales the pixel array via the hounsfield scale
# IMPORTANT: you must access value arr[y][x] to get the value at (x,y)
def loadImage(path):
  d = pd.dcmread(path)
  if hasattr(d, 'RescaleSlope') and hasattr(d, 'RescaleIntercept'):
    slope = float(d.RescaleSlope)
    intercept = float(d.RescaleIntercept)
    d.scaled_pixel_array = intercept + slope*np.copy(d.pixel_array)
  else:
    d.scaled_pixel_array = np.copy(d.pixel_array)
  debug(d)
  return d

# finds the x,y,r coordinate of the center of the image
def findCenterAndRadius(img):
  pixels = img.scaled_pixel_array
  width = len(pixels[0])
  height = len(pixels)
  debug(f'image is {width} x {height}')
 
  def findTop():
    for y in range(height):
      for x in range(width):
        if pixels[y][x] > 0:
          return (x,y)
  top = findTop()  # first positive pixel searching from left right, top down
  debug(f'top: {top}') # shuold be 238,38

  def findBottom():
    for y in range(height):
      for x in range(width):
        if pixels[height-1-y][x] > 0:
          return (x,height-1-y)
  bottom = findBottom()    # first positive pixel searching from left right, bottom up
  debug(f'bottom: {bottom}') # should be 239, 475

  def findLeft():
    for x in range(width):
      for y in range(height):
        if pixels[y][x] > 0:
          return (x,y)
  left = findLeft()    # first positive pixel searching from top down, left right
  debug(f'left: {left}') # shuold be 39, 245

  def findRight():
    for x in range(width):
      for y in range(height):
        if pixels[y][width-1-x] > 0:
          return (width-1-x,y)
  right = findRight()    # first positive pixel searching from top down, right left
  debug(f'right: {right}') # should be 477, 244

  x = round((right[0]+left[0])/2)
  y = round((bottom[1]+top[1])/2)
  r = y - top[1] - 25 # y coord of center - y coord of top is a fair estimate
  debug(f'center: ({x},{y}), radius: {r}')
  return (x,y,r)

# returns the slice that a point is in (0-indexed)
def getSliceNo(sliceWidth: float, distFromCenter: float) -> int:
  sliceNo = floor(distFromCenter/sliceWidth)
  return sliceNo


# calculates the number of sections a slice should have based on how big the slice is
def getNumSections(img, sliceNo: int, r: int, numSlices: int, maxSectionWidth) -> int:
  sliceWidth = r / numSlices
  r0 = (sliceNo + 1) * sliceWidth
  outerCircumference = pi * 2 * r0
  return floor(outerCircumference/maxSectionWidth)

# figures out which section in a slice a bearing falls
def getSectionNo(sliceNo: int, bearingFromCenter: float, sectionData) -> int:
  sections = sectionData[sliceNo]['sections']
  # search for the bearing
  for i in range(len(sections)):
    earlyAngle = sections[i]['bounds']['earlyAngle']
    lateAngle = sections[i]['bounds']['lateAngle']
    if (bearingFromCenter >= earlyAngle and bearingFromCenter <= lateAngle):
      return i
  debug(sliceNo, bearingFromCenter)
  raise Exception('section not found')

def getCenterpoint(
  innerRadius: int,
  outerRadius: int,
  earlyAngle: int,
  lateAngle: int,
  cx: int, cy: int,
) -> (int, int):
  theta = np.average([earlyAngle, lateAngle])
  r = np.average([innerRadius, outerRadius])
  # so start at (cx, cy), calculate the point that lies r euclidean distance away at an angle of theta
  coord = polarToRectangular(r, theta, cx, cy)
  return coord


# bins pixels into their sections
"""
sectionData = [
  {
    bounds: {
      innerRadius: float
      outerRadius: float
    mean: float
    sections: [
      {
        bounds: {
          earlyAngle: float
          lateAngle: float
        }
        pixels: [1, 3, 65, 2, 34, 45, 1 ...],
        mean: float
      ]
    }
  }
]
"""
def getSectionData(sectionData, img, numSlices: int, cx: int, cy: int, r: int, sectionWidth: int) -> None:
  sliceWidth = r/numSlices
  # populate section data
  # figure out boundaries for each
  for i in range(numSlices):
    numSections = getNumSections(img, i, r, numSlices, sectionWidth)
    sectionData.insert(i, {
      "bounds": {
        "innerRadius": sliceWidth * i,
        "outerRadius": sliceWidth * (i+1),
      },
      "sections": [],
    })

    for j in range(numSections):
      sectionData[i]["sections"].insert(j, {
        "bounds": {
          "earlyAngle": 360/numSections * j,
          "lateAngle": 360/numSections * (j+1),
        },
        "pixels": [],
      })

  pixels = img.scaled_pixel_array
  width = len(pixels)
  height = len(pixels[0])

  # loop through all the pixels and bin them
  for y in range(height):
    for x in range(width):
      distFromCenter = distance.euclidean((x,y), (cx, cy))
      if distFromCenter >= r:
        # outside the phantom, won't use
        continue
      else:
        if y == 112 and x == 266:
          noop()
        # see which slice its in
        sliceNo = getSliceNo(sliceWidth, distFromCenter)
        # now calculate angle
        bearingFromCenter = calculateBearing((cx, cy), (x,y))
        # see which section its in
        sectionNo = getSectionNo(sliceNo, bearingFromCenter, sectionData)
        val = pixels[y][x]
        sections = sectionData[sliceNo]['sections']
        sections[sectionNo]['pixels'].append(val)

# returns bearing in degrees
def calculateBearing(origin:(int,int), target:(int,int)) -> float:
  if origin == target:
    return 0
  def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

  def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

  north = (0,-1)
  vector = ( target[0] - origin[0] , target[1] - origin[1] )
  bearing = angle_between(north, vector) * 180 / pi
  if isnan(bearing):
    return 0
  if (vector[0] < 0):
    # the x coord is negative
    bearing = 360 - bearing
  return bearing

# print 2d array
def print2dArray(arr): 
  for r in arr:
    for c in r:
        print(c,end = " ")
    print()

# decides if a value is deviant given a tolerance
def isDeviant(val: float, tol: float) -> bool:
  return val < -tol

def polarToRectangular(r: float, theta: float, cx=0, cy=0) -> (int, int):
  x = r * cos((theta - 90) * pi / 180)
  y = r * sin((theta - 90) * pi / 180)
  return (round(x) + cx, round(y) + cy)