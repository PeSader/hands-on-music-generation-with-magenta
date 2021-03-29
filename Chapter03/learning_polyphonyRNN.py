import os
import math
import time
import typing

from importlib import import_module
from magenta.models import polyphony_rnn
from tensorflow.python.training.tracking.util import Checkpoint
from magenta.models.polyphony_rnn import polyphony_sequence_generator
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
    'polyphony': 'polyphony_rnn.mag',
}
# type: Dict[str, str]

MODEL_NAME_TO_SEQUENCE_GENERATOR = {
    'melody_rnn': 'melody_rnn_sequence_generator',
    'drums_rnn': 'drums_rnn_sequence_generator',
    'polyphony_rnn': 'polyphony_sequence_generator'
}
# type: Dict[str, str]

MODEL_NAME_TO_PACKAGING_METHOD = {
    'melody_rnn': 'bundle',
    'drums_rnn': 'bundle',
    'polyphony_rnn': 'bundle',
}


def get_bundle(bundle_name: str,
               bundle_dir: str = 'bundles',) -> GeneratorBundle:
    """ Downloads and reads specified magenta bundle

    Will first attempt to find the bundle by name, in the bundle_dir.  If not
    found, it will download it and then read it. Bundle files downloaded by
    this function will keep their official filenames.

    Args:
      bundle_name: str: Magenta pre-trained bundle, e.g. attention_rnn.mag
      bundle_dir: str: Target directory to download bundle files
        (Default value = 'bundles')

    Returns:
      GeneratorBundle: Bundle file that has already been read

    """
    from magenta.models.shared import sequence_generator_bundle

    bundle_file = os.path.join(bundle_dir, bundle_name)
    if not os.path.isfile(bundle_file):
        download_bundle(bundle_name, bundle_dir)
    bundle = sequence_generator_bundle.read_bundle_file(
        bundle_file=bundle_file)
    return bundle


def import_generator_module(model_name: str):
    """ Imports the sequence generator specific to the given model name

    Makes use of the importlib standard library to dinamically import sequence
    generator modules

    Args:
      model_name: str: Name of a magenta model, e.g. melody_rnn, drums_rnn

    Returns:
      Sequence generator module,
        e.g. magenta.models.melody_rnn.melody_rnn_sequence_generator


    """
    from importlib import import_module

    sequence_generator_module = MODEL_NAME_TO_SEQUENCE_GENERATOR[model_name]
    import_path = '.'.join(
        ['magenta.models',  model_name, sequence_generator_module])
    generator_module = import_module(import_path)
    return generator_module


def get_generator(generator_module,
                  generator_id: str,
                  checkpoint: Checkpoint,
                  bundle: GeneratorBundle):
    """ Returns an uninitialized generator object

        Such generator object is properly configured with the given
        generator_id, checkpoint, and bundle

    Args:
      generator_module: Sequence generator module
        (can be imported using import_generator_module())
      generator_id: str: A generator id, e.g. 'attention_rnn', 'drum_kit_rnn'
      checkpoint: Checkpoint: the checkpoint of a bundle
      bundle: GeneratorBundle: a generator bundle that has already been read

    Returns:
      An object whose class is inherited from BaseSequenceGenerator
        e.g. MelodyRnnSequenceGenerator, DrumsRnnSequenceGenerator

    Note:
      The sequence generator returned by this function has yet to be
        initialized using the .initialize() method

    """
    generator_map = generator_module.get_generator_map()
    generator = generator_map[generator_id](
        checkpoint=checkpoint, bundle=bundle)
    return generator


def get_primer_sequence(primer_file: str = None) -> NoteSequence:
    """ Converts primer midi file to NoteSequence

    Args:
      primer_file: str: Path to a midi file (Default value = None)

    Returns:
      NoteSequence: A note sequence obtained from the input file
        An empty NoteSequence is returned if no path is given

    """
    if primer_file:
        primer_sequence = midi_file_to_note_sequence(primer_file)
    else:
        primer_sequence = NoteSequence()
    return primer_sequence


def get_seconds_per_bar(tempo: int = DEFAULT_QUARTERS_PER_MINUTE) -> int:
    """Evaluates duration of a bar in seconds

    This is important because most magenta models take in duration in seconds,
    but music sounds off when an incomplete bar is played.

    Args:
      tempo: The tempo of the bar in quarters per minute.
        A quarter is a forth of the duration of a bar.
      tempo: int:  (Default value = DEFAULT_QUARTERS_PER_MINUTE)

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
    """ Evaluates the start and end times of sequence generation

    If a primer is given as a parameter and start_with_primer is set to True,
    the sequence generation will only begin after the primer has finished
    playing

    Args:
      generation_bars: int: duration, in bars, of the sequence to be generated
        (Default value = 1).
      primer_sequence: NoteSequence: a primer note sequence on which to base the
        tempo value, if given  (Default value = NoteSequence())
      start_with_primer: bool: True to take primer length into consideration,
        False otherwise (Default value = False)

    Returns:
      Dict[str, str]: a dict where start and end times of generation are stored
        in the keys 'start' and 'end', respectively

    Todo:
      * adjust time['end'] in such a way that the generated sequence does not
        end abruptly

    """
    if primer_sequence.tempos:
        primer_tempo = primer_sequence.tempos[0].qpm
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
    """ Setup GeneratorOptions object with given parameters

    This wrapper function facilitates setting up the GeneratorOptions object,
    which is common to all magenta models

    Args:
      time: dict: Contains generation start and end times,
        in keys 'start' and 'end'
      temperature: float: The greater is the temperature, the more random
        and more different from the primer is the resulting sequence
        (Default value = 1.0)
      beam_size: int: The greater is the beam_size, the longer is the sequence
        generated at each iteration (Default value = 1)
      branch_factor: int: The greater is the branch_factor, the more sequence
         candidates will be kept at each iteration (Default value = 1)
      steps_per_iteration: int: number of steps generated at each iteration
        (Default value = 1)

    Returns:
      GeneratorOptions: an object that contains the sequence generator options
        specified by the parameters of this function

    """
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
                      generation_length: int = 1,
                      qpm: float = DEFAULT_QUARTERS_PER_MINUTE,
                      temperature: float = 1.0,
                      beam_size: int = 1,
                      branch_factor: int = 1,
                      steps_per_iteration: int = 1,
                      primer_filename: str = '',
                      no_inject_primer_during_generation: bool = True,
                      condition_on_primer: bool = True,
                      bundle_dir: str = 'bundles',
                      ) -> NoteSequence:
    """ Generates a NoteSequence given (at least) a model and a generator id

    Uses model_name and generator_id to import the proper note sequence
    generator module as well as download the appropriate bundle or checkpoint
    file (which one depends on the chosen model).

    Args:
      model_name: str: Name of a magenta model, e.g. melody_rnn, drums_rnn
      generator_id: str: A pre-trained generator id, e.g. 'attention_rnn'
      checkpoint: Checkpoint: the checkpoint of a bundle
      generation_bars: int: duration, in bars, of the sequence to be generated
        (Default value = 1).
      temperature: float: The greater it is, the more random and more different
        from the primer is the resulting sequence (Default value = 1.0)
      beam_size: int: The greater is the beam_size, the longer is the sequence
        generated at each iteration (Default value = 1)
      branch_factor: int: The greater is the branch_factor, the more sequence
         candidates will be kept at each iteration (Default value = 1)
      steps_per_iteration: int: number of steps generated at each iteration
        (Default value = 1)
      primer_file: str: Path to a midi file (Default value = None)
      start_with_primer: bool: True to take primer length into consideration,
        False otherwise (Default value = False)
      bundle_dir: str: Target directory to download bundle files
        (Default value = 'bundles')

    Returns:
      NoteSequence: A note sequence in compliance to all given parameters

    Note:
      This function does not download the generated note sequence to a midi file
        use the function download_sequence() for that

    """

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
    primer_sequence = get_primer_sequence(primer_filename)

    # get generation start and end times
    time = get_generation_seconds(
        generation_length, primer_sequence, no_inject_primer_during_generation)
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
    """ Downloads a given NoteSequence to a local directory

    Args:
      model_name: str: Name of a magenta model, e.g. melody_rnn, drums_rnn
      generator_id: str: A generator id, e.g. 'attention_rnn', 'drum_kit_rnn'
      sequence: NoteSequence: the sequence to be downloaded as a midi file
        (Default value = NoteSequence())
      output_dir: str: Target directory to download resulting file
        (Default value = 'output')
    """
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
                             generation_length=2,
                             temperature=1.2,
                             primer_filename=primer_path,
                             no_inject_primer_during_generation=True,
                             )

download_sequence(model_name, generator_id, sequence)
