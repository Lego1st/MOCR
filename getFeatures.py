import cv2

def getFeatures(im, ratio):
  def getPe(side):
    pe = []
    avg = [] 
    for i in range(30):
      for j in range(30):
        if side == "left" and im[i][j] == 0:
          avg.append(j)
          break
        if side == "right" and im[i][29-j] == 0:
          avg.append(j)
          break
        if side == "top" and im[j][i] == 0:
          avg.append(j)
          break
        if side == "bot" and im[29-j][i] == 0:
          avg.append(j)
          break
      if (i+1)%3 == 0:
        if len(avg) < 3:
          avg = avg + [30]*(3-len(avg))
        if side == "left" or side == "right":
          pe.append((float)(sum(avg))/3)
        else:
          pe.append((float)(sum(avg))/3)
        avg = []
    return pe
  return getPe('left') + getPe('right') + getPe('top') + getPe('bot') + [ratio]
