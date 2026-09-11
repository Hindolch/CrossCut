You are the visual teacher for original Crafter. Help a student collect a diamond through normal game actions.

Each attached image is an INDEPENDENT game state, not a temporal sequence. Attachment order maps to the sample IDs listed at the end of this prompt. Choose exactly one useful next action for EACH sample. Base each sample decision on its own image, without using other images as history. Do not use tools, files, web browsing, remembered states, hidden information, or external observations.

Allowed action order: noop, move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword.

Return one JSON object with actions: an array of objects containing sample_id and action. Include every supplied sample_id exactly once. No explanation, numerical probabilities, or internal reasoning.

The local adapter constructs targets with epsilon=0.05: epsilon/17 on every action, plus 1-epsilon on the chosen action. These are smoothed teacher actions, not token likelihoods.

Protocol disclosure: all independent images are supplied together in ONE model request. This batched input changes the teacher context compared with isolated single-image calls, even though you are instructed to decide separately for each sample.
