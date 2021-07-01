import tello_code_execution
import AIRequests

CENTER = 0.5
pawnCoords, amountOfPawns, mostLeftPawn, centerPawn
connect()
send_command('takeoff')

while(True):
    pawnCoords = getCoords()
    amountOfPawns = len(pawnCoords)
    preferedPawn = None

    for coords in range(amountOfPawns):
        if(coords[0] < 0.45 ):
            direction = "left"
            if(coords[0] < mostLeftPawn[0] or mostLeftPawn is None):
                mostLeftPawn = coords
        elif(coords[0] > 0.45 && coords[0] < 0.65):
            direction = "center"
            if(abs(coords[0] - CENTER) < abs(centerPawn[0] - CENTER) or mostLeftPawn is None):
                centerPawn = coords
        else:
            direction = "left"
        
    if(amountOfPawns == 2):
        send_command('cw 1')
    elif(amountOfPawns == 1):
        if(direction == "right"):
                send_command('ccw 1')
            elif(direction == "left"):
                send_command('cw 1')
    elif(amountOfPawns == 0)
        send_command('land')
