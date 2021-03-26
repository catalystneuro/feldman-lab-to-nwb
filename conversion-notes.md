# Conversion notes for Feldman lab project.
# Questions are in braces []

Each NWB session is broken up into multiple ordered (ascending) "segments", indicated by the "\_S{\d+}\_" regular expression within each triple of (header, stimulus, trials) csv files.

## header.csv

Contains initial information for that segment, most critically the "stimulus layout" (StimLayout). Each different stimulus layout (of which there appear to be 8) produces a different Feldman lab "events" table, which can be represented as an NWB trials table - critically, this means each csv file can have a different structure in terms of field names and number of rows for each stimulus layout. It also contains the total number of different stimuli (Nstimuli) and type of stimulus orderering (StimOrder; integer index to "random", "ascending", "descending", "manual").

Also contains the starting and ending indices (from 0) of trials in that segment, as well as the ISS0time.

Most of this information maps to the "StimLookup" table, which contains the StimLayout integer and the 1:1 mapping to the text string StimType ({'Std','Trains','IL','Trains+IL','RFMap','2WC','MWS','MWD'}), as well as the number of stimuli and number of elements, but most importantly the "Elements" and "LogicalStim" tables.

The "Elements" table describes the piezo, piezolabel, amplitude, shape, etc. for each element. A "piezo", as far as I can infer, is an element within a stimulus presentation [there can be more than one for multiple whiskers?].

The "LogicalStim" table describes the makeup of each stimulus (its elements, timing, amplitude, GNG, etc.).

[In the shared data, all piezo labels are the same value, "--"; what does this mean?]
[What happens if stimulus:element ratio is not 1:1? How does mapping generalize?]
[Can we get nice text descriptions of these stimulus types?]
[need more example data for each layout to confirm mapping is OK]
[Can't find anywhere in the code where the A-Z row data is mapped?]

## stimulus.csv

For each stimulus layout, every row is a trial, and each column indicates the

1. "Posn": (ordinal position of this stimulus or element within the trial. (1=first stim in trial)
2. "StimElem": Used as the "events" table "ID", which is either an element number (if "events" type is an element) or stimulus number (if type is a logical stimulus) or trial number (if type is a trial).
3. "Time_ms": time in ms after trial onset
4. "Ampl": amplitude (microns for an element, or scaled fractional amplitude for a stimulus)

Thus I assume this file only represents "element" types from the "events" table.

## trials.csv

For stimulus layout 1, with no reward events, the trials.csv doesn't seem to correspond to any data in the "events" table. But does contain the important information for the trials table in NWB, such as start/end time.

[These may include invalid trials? Needs to be reconciled with the synch signals, etc.? As well as the ISS0 and arm times?]

[No reward information makes its way into the "events" table? Maybe just b/c the currently shared data doesn't have any type=4 values.]


## ToDo:

1. Creating a basic trials table from the start/end times, might be able to attach strings of stimulus layouts, etc.
2. Creating AnnotatedEvents table for other stimulus repetition information.
