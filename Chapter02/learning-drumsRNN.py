import os

from magenta.models.drums_rnn import drums_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq import constants
from note_seq.midi_io import note_sequence_to_midi_file
from note_seq.protobuf import generator_pb2
from note_seq import drums_lib
from note_seq.notebook_utils import download_bundle

# 1. Download the bundle. This snippet can later be turned into a function
download_bundle("drum_kit_rnn.mag", "bundles")
bundle = sequence_generator_bundle.read_bundle_file(
    os.path.join("bundles", "drum_kit_rnn.mag"))

# 2. Initialize the generator class using the drums_rnn model
# This and the next step can be refactored into a function
generator_map = drums_rnn_sequence_generator.get_generator_map()
generator = generator_map["drum_kit"](checkpoint=None, bundle=bundle)
generator.initialize()

# 3. Evaluate the length of a bar in seconds
qpm = 120
seconds_per_step = 60 / qpm / generator.steps_per_quarter
num_steps_per_bar = constants.DEFAULT_STEPS_PER_BAR
seconds_per_bar = num_steps_per_bar * seconds_per_step

# 4. Convert primer sequence to previously defined qpm
primer_drums = drums_lib.DrumTrack(
    [frozenset(pitches) for pitches in
     [(38, 51), (), (36,), (),
      (38, 44, 51), (), (36,), (),
      (), (), (38,), (),
      (38, 44), (), (36, 51), (), ]])
primer_sequence = primer_drums.to_sequence(qpm=qpm)

# 5. Since we want a one-bar-long primer, we define the following
primer_start_time = 0
primer_end_time = primer_start_time + seconds_per_bar

# 6. Calculate the the start and the end of the generated sequence
num_bar = 3
generation_start_time = primer_end_time
generation_end_time = generation_start_time + (seconds_per_bar * num_bar)

# 7. Configure the generator with start and end times
generator_options = generator_pb2.GeneratorOptions()
generator_options.args['temperature'].float_value = 1.1
generator_options.generate_sections.add(
    start_time=generation_start_time,
    end_time=generation_end_time)

# 8. Generate the sequence
sequence = generator.generate(primer_sequence, generator_options)

# 9. Write the resulting midi file to the output directory
midi_file = os.path.join("output", "drums_rnn.mid")
note_sequence_to_midi_file(sequence, midi_file)
