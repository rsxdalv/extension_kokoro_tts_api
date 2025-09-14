def to_wav_streaming_header(sample_rate, estimated_length=None):
    """
    Create a WAV header for streaming. If estimated_length is None,
    we'll use a placeholder that can be updated later.
    """
    import struct

    # WAV format parameters
    channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    # If we don't know the length, use a large placeholder
    if estimated_length is None:
        data_size = 0xFFFFFFFF - 36  # Max size minus header
    else:
        data_size = estimated_length * channels * bits_per_sample // 8

    # Create WAV header
    header = b"RIFF"
    header += struct.pack("<I", data_size + 36)  # File size - 8
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)  # PCM header size
    header += struct.pack("<H", 1)  # PCM format
    header += struct.pack("<H", channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", bits_per_sample)
    header += b"data"
    header += struct.pack("<I", data_size)

    return header
