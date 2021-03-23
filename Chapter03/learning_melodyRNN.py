import os
import math
import time
import typing

from importlib import import_module
from tensorflow.python.training.tracking.util import Checkpoint
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from note_seq.protobuf.generator_pb2 import GeneratorOptions, GeneratorBundle
from note_seq.protobuf.music_pb2 import NoteSequence
from note_seq.notebook_utils import download_bundle
from note_seq.midi_io import midi_file_to_note_sequence, note_sequence_to_midi_file

from note_seq.constants import(
    DEFAULT_QUARTERS_PER_MINUTE,
    DEFAULT_STEPS_PER_QUARTER,
    DEFAULT_STEPS_PER_SECOND,
    DEFAULT_STEPS_PER_BAR,
)

GENERATOR_ID_TO_BUNDLE_NAME = {
    'basic_rnn': 'basic_rnn.mag',
    'lookback_rnn': 'lookback_rnn.mag',
    'attention_rnn': 'attention_rnn.mag',
}
# type: Dict[str, str]

MODEL_NAME_TO_SEQUENCE_GENERATOR = {
    'melody_rnn': 'melody_rnn_sequence_generator',
    'drums_rnn': 'drums_rnn_sequence_generator',
}
# type: Dict[str, str]


def get_bundle(bundle_name: str,
               bundle_dir: str = 'bundles',) -> GeneratorBundle:
    from magenta.models.shared import sequence_generator_bundle

    bundle_file = os.path.join(bundle_dir, bundle_name)
    if not os.path.isfile(bundle_file):
        download_bundle(bundle_name, bundle_dir)
    bundle = sequence_generator_bundle.read_bundle_file(
        bundle_file=bundle_file)
    return bundle


def import_generator_module(model_name: str):
    sequence_generator_module = MODEL_NAME_TO_SEQUENCE_GENERATOR[model_name]
    import_path = '.'.join(
        ['magenta.models',  model_name, sequence_generator_module])
    generator_module = import_module(import_path)
    return generator_module


def get_generator(generator_module,
                  generator_id: str,
                  checkpoint: Checkpoint,
                  bundle: GeneratorBundle):
    generator_map = generator_module.get_generator_map()
    generator = generator_map[generator_id](
        checkpoint=checkpoint, bundle=bundle)
    return generator


def get_primer_sequence(primer_file: str = None) -> NoteSequence:
    if primer_file:
        primer_sequence = midi_file_to_note_sequence(primer_file)
    else:
        primer_sequence = NoteSequence()
    return primer_sequence


def get_seconds_per_bar(tempo: int = DEFAULT_QUARTERS_PER_MINUTE) -> int:
    """
    Evaluate duration of a bar in seconds

    This is important because most magenta models take in duration in seconds,
    but music sounds off when an incomplete bar is played.

    Args:
        tempo: The tempo of the bar in quarters per minute.
            A quarter is a forth of the duration of a bar.

    Returns:
        float: Number of seconds per bar

    """
    SECONDS_PER_MINUTE = 60
    STEPS_PER_QUARTER = DEFAULT_STEPS_PER_QUARTER
    STEPS_PER_BAR = DEFAULT_STEPS_PER_BAR

    seconds_per_bar = (SECONDS_PER_MINUTE / tempo /
                       STEPS_PER_QUARTER) * STEPS_PER_BAR
    return seconds_per_bar


def get_generation_seconds(generation_bars: int = 1,
                           primer_sequence: NoteSequence = NoteSequence(),
                           start_with_primer: bool = False,
                           ) -> typing.Dict[str, str]:
    if primer_sequence:
        primer_tempo = primer_sequence.tempos[0].qpm
        seconds_per_bar = get_seconds_per_bar(primer_tempo)
    else:
        seconds_per_bar = get_seconds_per_bar()
    time = {}
    if start_with_primer:
        time['start'] = primer_sequence.total_time
    else:
        time['start'] = 0
    time['end'] = time['start'] + generation_bars * seconds_per_bar
    return time


def setup_generator_options(time: dict,
                            temperature: float = 1.0,
                            beam_size: int = 1,
                            branch_factor: int = 1,
                            steps_per_iteration: int = 1,
                            ) -> GeneratorOptions:
    generator_options = GeneratorOptions()
    generator_options.args['temperature'].float_value = temperature
    generator_options.args['beam_size'].int_value = beam_size
    generator_options.args['branch_factor'].int_value = branch_factor
    generator_options.args['steps_per_iteration'].int_value = steps_per_iteration
    generator_options.generate_sections.add(
        start_time=time['start'],
        end_time=time['end'])
    return generator_options


def generate_sequence(model_name,
                      generator_id,
                      checkpoint: Checkpoint = None,
                      generation_bars: int = 1,
                      temperature: float = 1.0,
                      beam_size: int = 1,
                      branch_factor: int = 1,
                      steps_per_iteration: int = 1,
                      primer_file: str = None,
                      start_with_primer: bool = False,
                      bundle_dir: str = 'bundles',
                      ) -> NoteSequence:

    # download and read bundle files
    bundle_name = GENERATOR_ID_TO_BUNDLE_NAME[generator_id]
    bundle = get_bundle(bundle_name, bundle_dir)

    # dinamically import the sequence generator module given the model name
    generator_module = import_generator_module(model_name)

    # initialize the generator
    generator = get_generator(
        generator_module, generator_id, checkpoint, bundle)
    generator.initialize()

    # convert the primer midi file to a NoteSequence
    primer_sequence = get_primer_sequence(primer_file)

    # get generation start and end times
    time = get_generation_seconds(
        generation_bars, primer_sequence, start_with_primer)
    # configure generator options with given parameters
    generator_options = setup_generator_options(time, temperature, beam_size,
                                                branch_factor, steps_per_iteration)
    # generate the note sequence
    sequence = generator.generate(primer_sequence, generator_options)
    return sequence


def download_sequence(model_name: str,
                      generator_id: str,
                      sequence: NoteSequence = NoteSequence(),
                      output_dir: str = 'output',
                      ) -> None:
    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    midi_filename = "%s_%s_%s.mid" % (model_name, generator_id, date_and_time)
    midi_file = os.path.join(output_dir, midi_filename)
    note_sequence_to_midi_file(sequence, midi_file)


primer_dir = 'primers'
primer_filename = 'Fur_Elisa_Beethoveen_Monophonic.mid'
primer_path = os.path.join(primer_dir, primer_filename)

model_name = 'melody_rnn'
generator_id = 'attention_rnn'

sequence = generate_sequence(model_name=model_name,
                             generator_id=generator_id,
                             checkpoint=None,
                             generation_bars=2,
                             temperature=1.2,
                             primer_file=primer_path,
                             start_with_primer=True,
                             )

download_sequence(model_name, generator_id, sequence)
