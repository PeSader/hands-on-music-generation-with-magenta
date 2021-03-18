import os

import magenta.music as mm
from magenta.models.drums_rnn import drums_rnn_sequence_generator
from magenta.music import constants

# 1. Download the bundle. This snippet can later be turned into a function
mm.notebook_utils.download_bundle("drum_kit_rnn.mag", "bundles")
bundle = mm.sequence_generator_bundle.read_bundle_file(
    os.path.join("bundles", "drum_kit_rnn"))

# 2. Initialize the generator class using the drums_rnn model This and the next step can be refactored into a function
generator_map = drums_rnn_sequence_generator.get_generator_map()
generator = generator_map["drum_kit"](checkpoint=None, bundle=bundle)
generator.initialize()

# 3. Evaluate the length of a bar in seconds
qpm = 120
seconds_per_step = 60 / qpm / generator.steps_per_quarter
num_steps_per_bar = constants.DEFAULT_STEPS_PER_BAR
seconds_per_bar = num_steps_per_bar * seconds_per_step
