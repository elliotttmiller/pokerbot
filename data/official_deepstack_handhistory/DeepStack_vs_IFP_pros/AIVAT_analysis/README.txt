This directory contains summary data for each participant's AIVAT analysis, formatted as a .csv (comma-separated values) spreadsheet. It can be opened as plain text using a basic text editor like Notepad on Windows or TextEdit on OSX. However, you will likely want to use your preferred spreadsheet software (e.g., Microsoft Excel, Numbers, Google Sheets, OpenOffice, LibreOffice).

The .csv files are formatted as follows. Each line of the file corresponds to a single hand from the participant's match against DeepStack, ordered by the time they started the hand (like the associated match log files.)  The first line of the file provides labels that correspond to each column/field of data in the table. We introduce each field's meaning below.

AIVAT: how many of DeepStack's chips the participant was expected to take in this hand, adjusted for luck. This is the most important number, and is the final score used to evaluate each hand. It is equal to the "all hands chips" value, minus the two luck correction terms (described below).

Chips: how many of DeepStack's chips the participant took in this hand.

All Hands Chips: how many of DeepStack's chips the participant took, playing against DeepStack's range. This value looks at all possible hands DeepStack could have held, and how likely they were.

Chance Correction: an estimate of how lucky the participant's cards and the board cards were, in chips. For example, if the chance correction was -117.2, the participant was expected to win 117.2 more chips than the all hands chips value.

Action Correction: an estimate of how lucky DeepStack's actions were for the participant, in chips. For example, if the chance correction was -40.8, the action DeepStack picked worked out more poorly for the participant than the other actions it could have taken, and the participant expected to win 40.8 more chips than the all hands chips value.

Note that the above values are from the participant's point of view: a negative correction term means they got unlucky, and negative AIVAT/Chips values mean they lost chips.

Next, there are several fields that describe what happened in the hand including the cards dealt ("Your Hand", "DeepStack Hand", "Flop Cards", "Turn Card", "River Card"), the participant's "Position" (big blind versus small blind), and each round's betting ("Pre-flop", "Flop", "Turn", and "River"). If a given betting round was not reached due to a player folding, that round's corresponding public cards and betting are left empty. The betting fields use a similar format to the one used for the ACPC-formatted match logs.
 'f' : folds
 'c' : check/call
 'rX' : raise to X  (note, "raise to" rather than "raise by": the player's total chips in the pot)
 Note that the initial 50/100 blinds are not listed and all of the actions by the big blind are capitalised.


Finally, there are some auxiliary fields regarding the hand.

Seconds to Act: Time taken by the participant's actions. Note that this includes any network lag, and delay from the browser interface.

Total Seconds: Total time taken for the hand, including the participant's actions and DeepStack's actions.

Start Time (UTC): The (coordinated universal) time that the hand started.

Notes: Any special notes about the hand. In particular, DeepStack disconnected on nine hands against two participants, causing it to check/fold against them for the rest of the hand.
