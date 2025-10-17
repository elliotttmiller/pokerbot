This directory contains DeepStack match logs in the format used for the Annual Computer Poker Competition (with an added timestamp).  The files are formatted as plain text with each line corresponding to a game/hand of poker.  You should be able to open the file with your choice of text editor.  Some options that should come included on your computer are Notepad for Windows or TextEdit for OSX.  

The high-level format of each hand is:

STATE:<Hand#>:<Betting>:<Cards>:<Payoff>:<Player Positions> # <Timestamp>

We use the following example hand to illustrate.

STATE:19:r200c/cr400c/cr850f:9hQc|5dQd/4h4c6c/4s:-400|400:DeepStack|player.account # 1479132064.35509


"STATE:19" – Hand number since table start

Note that hand numbers were reset to 0 when participants started a new table (either for multitabling or if they left the table to resume playing later).


"DeepStack|player.account" – Player positions

In this example DeepStack is player 1, the big blind, and player 2 (the participant) would be in the small blind.


"r200c/cr400c/cr850f" – The betting history

 'f' : folds
 'c' : check/call
 'rX' : raise to X  (note, "raise to" rather than "raise by": the player's total chips in the pot)
 '/' : round separator
 Note that the initial 50/100 blinds are not listed.


"9hQc|5dQd/4h4c6c/4s" – Cards dealt

The private cards for each player, starting with the big blind (player 1, 9hQc), the small blind (player 2, 5dQd), and the public cards for each round (separated with '/').


-400|400 – Chips won/lost by player1|player2


"# 1479132064.35509" – Unix timestamp for when the hand started

Hands have been sorted chronologically according to this time.
