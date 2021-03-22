from note_seq import NoteSequence
import os
import typing

from magenta.models.drums_rnn import drums_rnn_sequence_generator
from magenta.models.drums_rnn.drums_rnn_sequence_generator import (
    DrumsRnnSequenceGenerator,
)
from magenta.models.shared import sequence_generator_bundle
from note_seq import constants
from note_seq import drums_lib
from note_seq.midi_io import note_sequence_to_midi_file
from note_seq.notebook_utils import download_bundle
from note_seq.protobuf import generator_pb2

DRUM_KIT_RNN_BUNDLE_NAME = 'drum_kit_rnn.mag'
DRUMS_KIT_RNN_GENERATOR_ID = 'drum_kit'


def get_drums_generator(bundle,) -> DrumsRnnSequenceGenerator:
    """
    Initializes a drums_rnn generator, not taking a checkpoint as an argument

    Returns:
        DrumsRnnSequenceGenerator: a sequence generator for the drums_rnn model


    """

    generator_map = drums_rnn_sequence_generator.get_generator_map()
    generator = generator_map[DRUMS_KIT_RNN_GENERATOR_ID](
        checkpoint=None, bundle=bundle)
    generator.initialize()
    return generator


def get_seconds_per_bar(tempo: int = constants.DEFAULT_QUARTERS_PER_MINUTE,
                        ) -> int:
    """
    Evaluate duration of a bar in seconds

    This is important because most magenta models take in duration in seconds,
    but music sounds off when an incomplete bar is played.

    Args:
        tempo: The tempo of the bar in quarters per minute.
            A quarter is a quarter of a bar.

    Returns:
        int: Number of seconds per bar

    """
    SECONDS_PER_MINUTE = 60
    STEPS_PER_QUARTER = constants.DEFAULT_STEPS_PER_QUARTER
    STEPS_PER_BAR = constants.DEFAULT_STEPS_PER_BAR

    seconds_per_bar = (SECONDS_PER_MINUTE / tempo /
                       STEPS_PER_QUARTER) * STEPS_PER_BAR
    return seconds_per_bar


def get_drums_bundle(bundle_dir: str = 'bundles'):
    bundle_file = os.path.join(bundle_dir, DRUM_KIT_RNN_BUNDLE_NAME)
    if not os.path.isfile(bundle_file):
        download_bundle(DRUM_KIT_RNN_BUNDLE_NAME, 'bundles')
    bundle = sequence_generator_bundle.read_bundle_file(
        bundle_file=bundle_file)
    return bundle


def get_generation_times(generation_length: int = 1,
                         start_with_primer: bool = False,
                         ) -> typing.Dict[str, str]:
    seconds_per_bar = get_seconds_per_bar()
    time = {}
    if start_with_primer:
        time['start'] = seconds_per_bar  # one-bar-long primer
    else:
        time['start'] = 0
    time['end'] = time['start'] + generation_length * seconds_per_bar
    return time


def generate_drums(bundle_dir: str = 'bundles',
                   target_dir: str = 'output',
                   generation_length: int = 1,
                   temperature: float = 1.1,
                   input_sequence: NoteSequence = None,
                   start_with_primer: bool = False,
                   ) -> None:
    """Generates a drums midi file

    Generates a midi file of drums percussion, using the pre-trained
    drum_kit_rnn magenta model.

    Args:
        bundle_dir: The directory where the bundle will be downloaded to.
        generation_length: Duration in bars of the generated sequence.
        temp: The degree of randomness of the generation.
        primer_sequence: Optional, to "influence" the generation.
        start_with_primer: True for primer sequence to play before the
            generated one, False to not play the primer sequence.

    Note:
        Primer must be at least one-bar-long if start_with_primer is True

    """
    generator = get_drums_generator(
        bundle=get_drums_bundle(bundle_dir=bundle_dir))
    time = get_generation_times(generation_length=generation_length,
                                start_with_primer=start_with_primer)

    generator_options = generator_pb2.GeneratorOptions()
    generator_options.args['temperature'].float_value = temperature
    generator_options.generate_sections.add(
        start_time=time['start'],
        end_time=time['end'])

    sequence = generator.generate(input_sequence, generator_options)
    note_sequence_to_midi_file(
        sequence, os.path.join(target_dir, 'drums_rnn.mid'))


primer_drums = drums_lib.DrumTrack(
    [frozenset(pitches) for pitches in
     [(38, 51), (), (36,), (),
      (38, 44, 51), (), (36,), (),
      (), (), (38,), (),
      (38, 44), (), (36, 51), (), ]])
primer_sequence = primer_drums.to_sequence(
    qpm=constants.DEFAULT_QUARTERS_PER_MINUTE)

generate_drums(generation_length=5, temperature=1.2,
               input_sequence=primer_sequence, start_with_primer=True)
