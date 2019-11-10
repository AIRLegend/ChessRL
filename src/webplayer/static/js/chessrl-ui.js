var board1 = null

function initBoard() {
    board1 = Chessboard('board1', {
        draggable: true,
        onDrop: onDrop
    })

    board1.start()

    $.ajax({
        type: "GET",
        url: 'http://localhost:5000/game',
        success: serverResponse
    })

    $("#reset-btn").click(resetGame)
    $("#switch-btn").click(changeColor)
}


function onDrop (source, target, piece, newPos, oldPos, orientation) {
    $.ajax({
        type: "POST",
        url: 'http://localhost:5000/game/' + source+target,
        success: serverResponse
    })
}




function resultToString(result) {
    string = "In progress"
    switch(result){
        case 0:
            string = "Draw"
            break
        case -1:
            string = "Blacks win"
            break
        case 1:
            string = "Whites win"
            break
        default:
            string = "In progress"
            break
    }
    return string

}

function serverResponse(data) {
    // Update board
    board1.position(data['fen'], false)

    //Disable board if end game
    if (data['hist']['result'] != null){
        board1.draggable = false
    }

    // Game status
    resStr = resultToString(data['hist']['result'])
    $("#result-span").text("Game state: " +resStr)

    // Game date
    $("#game-date").text("Game date: " + data['hist']['date'])

    // Update table of moves
    $("#table-body").empty()
    moves = data["hist"]["moves"]
    for(i=0; i<moves.length; i++){
        color = i==0 || i%2 == 0 ? "<td>Whites</td>" : "<td>Blacks</td>"
        m = "<td>"+moves[i]+"</td>"
        html = '<tr> <th scope="row">'+i+'</th>'+color+m +'</tr>'
        $("#table-body").append(html)
    }

}

function changeColor() {
    if (board1.orientation() == "white") {  // Change to black
       resetGame(0)
    } else {  // Change to white
       resetGame(1)
    }
    board1.orientation('flip')
}

/**
 * Deletes the current game on the server and request other.
 * @param color. 1 If we want to play as white, 0 if not
 */
function resetGame(color = 0){
    $.ajax({
        type: "PUT",
        url: 'http://localhost:5000/game/'+color
    }).done(res => $.ajax({
            type: "GET",
            url: 'http://localhost:5000/game',
            success: serverResponse
        }))
}


