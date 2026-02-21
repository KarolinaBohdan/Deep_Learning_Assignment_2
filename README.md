Human In The Loop

After LLM labeling I inspected the file and found that LLM labled wrong 5 examples. 

For example: 

1. doc_id 9638427, Apparatus for substantially blocking flames and spreading heated gases from a broiler flue, said apparatus comprising a riser adapted to be placed above the broiler flue for defining a pathway along which heated gases are exhausted from the flue, a catalyst support on the riser for supporting a catalyst in the pathway for flow of heated gases through the catalyst, and a plurality of baffles in the riser extending across the pathway below the catalyst support, said baffles having the shape of inverted troughs for substantially blocking flames emitted from the broiler flue and for spreading heated gases exhausted from the flue for more uniform distribution of the heated gases across a bottom surface of the catalyst, the plurality of baffles comprising a first baffle, a second baffle, and a third baffle, wherein the second baffle is centered between the first and third baffles, wherein the second baffle is spaced from the first baffle and the third baffle by a horizontal distance D wherein the second baffle is larger in cross-sectional height and width than the first and third baffles. -- LLM gave 0.0 (NOT_GREEN), The correct label should be 1.0 (GREEN)
2. doc_id 8948831, Transmission system comprising: a superconductive cable having three phase conductors; and a cryostat surrounding the phase conductors and enclosing a hollow space for guiding a cooling agent, wherein a neutral conductor, common to all three phase conductors, is provided, and wherein the cryostat includes a circumferentially enclosed, thermally insulated sheath, wherein the neutral conductor is an electrically normally conducting material and is arranged outside of the cryostat, and the neutral conductor is constructed as an insulated round conductor and is placed in a separate protective pipe. -- LLM labeled as 0.0 (NOT_GREEN), should be 1.0 (GRREN)

The results from before fine-tuning:
Precision=0.7916 - Recall=0.7726 - F1=0.7820

After fine-tuning:
Precesion=0.8341 - Recall=0.7971 - F1=0.81525

We then trained the model on AI-lab and got 
