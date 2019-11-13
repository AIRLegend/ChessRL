var board1 = null

var lastMove = []

function initBoard() {
    board1 = Chessboard('board1', {
        draggable: true,
        onDrop: onDrop
    })

    board1.start()

    $.ajax({
        type: "GET",
        url: 'http://localhost:5000/game',
        success: handleServerResponse
    })

    $("#reset-btn").click(handleResetGame)
    $("#switch-btn").click(handleChangeColor)
    $("#button-promotion").click(handlePromotionForm)
}


function onDrop (source, target, piece, newPos, oldPos, orientation) {

    lastMove = [source, target] //Save the move

    promotion = checkIfPromotion(source, target, piece)

    if (promotion) {
        // Show UI choose promotion
        togglePromotionForm(true)
        // The form handler will be responsible of sending the move
    } else {
        togglePromotionForm(false)
        sendMove(source, target)
    }
    return

    
}

function sendMove(source, target, promotion=null) {
    if (promotion !== null) {
        target += promotion
    }

    $.ajax({
        type: "POST",
        url: 'http://localhost:5000/game/' + source+target,
        success: handleServerResponse
    })
}



/**
 * Checks if a Pawn has reached the other board side
 */
function checkIfPromotion(source, target, piece) {
    promotion = false
    if (piece.endsWith('P')) {
        if (piece.startsWith('w')) {
            if (target.endsWith('8')) {
                // Deserves a promotion
                promotion = true
            }
        } else if (piece.startsWith('b')){
            if (target.endsWith('1')) {
                // Deserves a promotion
                promotion = true
            }
        }
    }
    return promotion
}

/**
 * shows or hides the promotion form
 * @param visible 
 */
function togglePromotionForm(visible=false) {
    if (visible) {
        $("#promotion").show()
    } else {
        $("#promotion").hide()
    }
}


/**
 * Converts the API result to a readable string
 * @param {*} result 
 */
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



/* HANDLERS */

/**
 * Gets the response of the user to the promotion form selection
 * (when he clicks OK) and send the UCI move to the API, which is
 * defined as move+promotion_piece.
 * Also it hides the form.
 * 
 * It is not necessary to update the piece in the UI. The server 
 * response will update it.
 */
function handlePromotionForm() {
    selected = $("#promotion-select option:selected").val()
    togglePromotionForm(false)
    sendMove(lastMove[0], lastMove[1], selected)
}

function handleChangeColor() {
    if (board1.orientation() == "white") {  // Change to black
        handleResetGame(0)
    } else {  // Change to white
        handleResetGame(1)
    }
    board1.orientation('flip')
}

/**
 * Deletes the current game on the server and request other.
 * @param color. 1 If we want to play as white, 0 if not
 */
function handleResetGame(color = 0){
    $.ajax({
        type: "PUT",
        url: 'http://localhost:5000/game/'+color
    }).done(res => $.ajax({
            type: "GET",
            url: 'http://localhost:5000/game',
            success: handleServerResponse
        }))
}

/**
 * Handles the response of the server updating the UI.
 * @param data 
 */
function handleServerResponse(data) {
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
        color = i==0 || i%2 == 0 ? "<td>Whites</td>" : "<td>Blacks</td>"
        m = "<td>"+moves[i]+"</td>"
        html = '<tr> <th scope="row">'+i+'</th>'+color+m +'</tr>'
        $("#table-body").append(html)
    }

}
