
import cv2 , time
import datetime




def saveImg( image , frame ):
    d = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")

    filename = 'd:/work/camLog/' + d + '_' + str(frame) + '.png'
    print(filename)
    cv2.imwrite( filename , image)


def printTime( prefix , t) :
    print( prefix + '{0:.5f}'.format(t))


def checkPic( a , b , inverse=False ) :
    global lastPic
    global hits
    global threshold

    if inverse:
        diff = cv2.subtract( b, a )
        cv2.imshow("b-a", diff)
    else:
        diff = cv2.subtract( b, a )
        cv2.imshow("a-b", diff)

    mean = cv2.mean(diff)[0]
    dbgOut(mean)

    if mean > threshold:
        hits += 1
        #print('cpy last')
        print(  str(hits) + ' :: ' +  '{0:.3f}'.format(mean) )
        lastPic = a
        return True

    return False

def checkThis( a , b , o , h):
    global logEnable

    if checkPic(a, b, inverse=False):
        if logEnable:
            saveImg(image=o, frame=h)
        # else:
        #    print('a-b hit: ' + str(hits))
    else:
        if checkPic(a=a, b=b, inverse=True):
            if logEnable:
                saveImg(image=o, frame=h)
            # else:
            #    print('b-a hit: ' + str(hits))

			
def dbgOut( s ):
    global debug
    
    if debug:
        print(s)

def logChanges():

    global logEnable
    global threshold
    global lastPic
    global hits
    global debug
    global img_width
    global img_height

    gain =64
    contrast=127
    brightness=1
	
    cam = cv2.VideoCapture(0)
    print('got capture')

    cam.set(3, img_width)     # width
    cam.set(3, img_height) # height

    #cam.set(11, contrast)  # contrast       min: 0   , max: 255 , increment:1
    #cam.set(14, gain)  # gain           min: 0   , max: 127 , increment:1

    #cam.set(15, -1)  # exposure       min: -7  , max: -1  , increment:1


    #cam.set(10, 120)  # brightness     min: 0   , max: 255 , increment:1
    #cam.set(12, 70)  # saturation     min: 0   , max: 255 , increment:1
    #cam.set(13, 13)  # hue
    #cam.set(17, 5000)  # white_balance  min: 4000, max: 7000, increment:1
    #cam.set(28, 0)  # focus          min: 0   , max: 255 , increment:5

    check, original = cam.read()
    #rows = int(str(len(original)))
    #cols = int(str(len(original[0])))
    #print('size:  ' + str(rows) + 'x' + str(cols) )

    print('Paused  ... press space to proceed')

    frame = 0
    thresholdInc = 0.01
	
    print('Press 1|2 to change violation threshold= ' + str(threshold))

    while True:

        runStart=time.time()
        check, original = cam.read()
        cv2.imshow("original", original)

        thisPic = cv2.cvtColor( original , cv2.COLOR_BGR2GRAY)
        #cv2.imshow( "thisPic" , thisPic)


        if frame > 0:
            checkThis( thisPic , lastPic , original , hits )

        else:
            lastPic = thisPic

        frame += 1


        #press any key to out
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        if key == ord(' '):
            logEnable = not logEnable
            if logEnable:
                print('Logging ... press space to pause')
            else:
                print('Paused  ... press space to proceed')

        if key == ord('1'):
            threshold-=thresholdInc
            print("treshold: {0:.5f}".format(threshold))
        if key == ord('2'):
            threshold+=thresholdInc
            print("treshold: {0:.5f}".format(threshold))
        if key == ord('3'):
            debug = not debug
            print("debug: " + str(debug))
        
        #if key == ord('4'):
        #    if contrast > 0:
        #        contrast -= 1 
        #        print('contrast:'+str(contrast) )
        #        cam.set(11, contrast)  # contrast       min: 0   , max: 255 , increment:1
        #if key == ord('5'):
        #    if contrast <255:
        #        contrast += 1 
        #        print('contrast:'+str(contrast) )
        #        cam.set(11, contrast)  # contrast       min: 0   , max: 255 , increment:1
        #if key == ord('4'):
        #    if gain > 0:
        #        gain -= 1 
        #        print('gain:'+str(gain) )
        #        cam.set(14, gain)  # gain       min: 0   , max: 127 , increment:1
        #if key == ord('5'):
        #    if gain <255:
        #        gain += 1 
        #        print('gain:'+str(gain) )
        #        cam.set(14, gain)  # gain       min: 0   , max: 127 , increment:1

        if key == ord('4'):
            if brightness > 0:
                brightness -= 1 
                print('brightness:'+str(brightness) )
                cam.set(10, brightness)  # brightness       min: 0   , max: 255 , increment:1
        if key == ord('5'):
            if brightness <255:
                brightness += 1 
                print('brightness:'+str(brightness) )
                cam.set(11, brightness)  # brightness       min: 0   , max: 255 , increment:1

			


        #printTime('\nframerate: ' , (1/(time.time()-runStart) ))

    #shutdown
    cam.release()
    cv2.destroyAllWindows()


#logChanges()
logEnable = False
threshold = 1.72
lastPic=0
hits =0
debug = False

img_width=320
img_height=240

if __name__ == "__main__":
    logChanges()
