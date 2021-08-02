#!/usr/bin/env python3

import math
import numpy as np

from data import *


class StreamState:
    def __init__(self):
        self.comp_state = np.asarray([0] * 320)


stream_state = StreamState()

total_bits = 16000 // 50
freq_bins = 14


class BitStream:
    def __init__(self, word_size=16):
        self.acc = 0
        self.bits = 0
        self.word_size = word_size

    def push(self, value, width):
        self.bits += width
        self.acc <<= width
        assert value >= 0
        assert value < (1 << width)
        self.acc |= value

    def tolist(self):
        bits = self.bits
        acc = self.acc

        while bits % self.word_size:
            bits += 1
            acc <<= 1

        ret = []
        while acc:
            ret.append(acc & ((1 << self.word_size) - 1))
            acc >>= self.word_size
        ret.reverse()
        return ret

    def complete_words(self):
        return self.bits // self.word_size


def a1800_enc_frame(decompressed):
    decompressed = np.asarray(decompressed)
    some_number_of_bits, freq = enc_frame_1(decompressed, 320)
    compressed = enc_frame_2(freq, some_number_of_bits)
    return compressed


def enc_frame_1(decompressed, length):
    S = stream_state.comp_state
    LT = (18318 * np.sin(math.pi * (np.arange(0, 320) + 0.5) / 640)).astype(int)
    IN = decompressed
    L = length
    HL = L // 2
    LT1 = LT[:HL]
    LT2 = LT[HL:]
    S1 = S[:HL]
    S2 = S[HL:]
    IN1 = IN[:HL]
    IN2 = IN[HL:]

    A1 = np.concatenate((np.flip(LT1), np.flip(LT2)))
    B1 = np.concatenate(( np.flip(S1),          IN1))

    A2 = np.concatenate((         LT2,         -LT1))
    B2 = np.concatenate((          S2, np.flip(IN2)))
    T = (A1 * B1 + A2 * B2) // 32768

    stream_state.comp_state = decompressed
    sVar3 = np.max(np.abs(T))
    some_number_of_bits = calc_number_of_bits(sVar3)
    uVar12 = np.sum(np.abs(T))
    if sVar3 < uVar12 // 128:
        some_number_of_bits -= 1

    # scale signal so that the maximum is ~10k
    if some_number_of_bits < 0:
        T = T // (2 ** (-some_number_of_bits))
    elif some_number_of_bits > 0:
        T = T * (2 ** some_number_of_bits)

    freq = time_to_freq_domain(T)

    return some_number_of_bits, freq


def calc_number_of_bits(x):
    if x >= 14000:
        return 0
    if x < 54:
        return 9
    if x < 438:
        x += 1
    return 8 - int(math.log(x * 19174 // 1048576, 2))


def time_to_freq_domain_1(Tin, j):
    L = len(Tin)
    Tout = []
    for i in range(1 << j):
        a = i * (L >> j)
        b = L >> j
        A = Tin[a : a + b - 1 : 2] // 2 + Tin[1 + a : 1 + a + b - 1 : 2] // 2
        B = np.flip(Tin[a : a + b - 1 : 2] // 2 - Tin[1 + a : 1 + a + b - 1 : 2] // 2)
        Tout = np.concatenate((Tout, A, B))
    return Tout


def time_to_freq_domain_2(Tin, LT, j):
    L = len(Tin)
    Tout = []
    for i in range(1 << j):
        a = i * (L >> j)
        c = (L >> j) // 2
        A = (LT[ ::4] * Tin[  a:  a+c-1:2] - LT[1::4] * Tin[  a+c:  a+2*c-1:2]) // 32768
        B = (LT[2::4] * Tin[1+a:1+a+c-1:2] + LT[3::4] * Tin[1+a+c:1+a+2*c-1:2]) // 32768
        C = (LT[1::4] * Tin[  a:  a+c-1:2] + LT[ ::4] * Tin[  a+c:  a+2*c-1:2]) // 32768
        D = (LT[3::4] * Tin[1+a:1+a+c-1:2] - LT[2::4] * Tin[1+a+c:1+a+2*c-1:2]) // 32768
        T = np.zeros((2 * c))
        T[0:c:2] = A
        T[1:c:2] = B
        T[c + 1 : 2 * c + 1 : 2] = np.flip(C)
        T[c : 2 * c : 2] = np.flip(D)
        Tout = np.concatenate((Tout, T))
    return Tout


def time_to_freq_domain(T):
    T1 = np.asarray(SHORT_ARRAY_1000bb88)

    for j in range(5):
        T = time_to_freq_domain_1(T, j)

    T2 = np.zeros((32, 10))
    for k in range(32):
        for j in range(10):
            T2[k, j] = 2 * np.dot(T1[j], T[10 * k : 10 * k + 10]) // 65536

    freq = np.reshape(T2, -1)
    for j in range(4, -1, -1):
        freq = time_to_freq_domain_2(freq, PTR_ARRAY_1000bb70[4 - j], j)

    return freq


def enc_frame_2(freq, some_number_of_bits):
    data_out2_bits, data_out1, data_out1_bits = unknown1(freq, some_number_of_bits)
    available_bits = total_bits - np.sum(data_out1_bits[:freq_bins]) - 4
    bit_table1, bit_table2 = unknown2(available_bits, 16, data_out2_bits)
    data_out2_bits = data_out2_bits + some_number_of_bits * 2 + 24
    data_out2_bits, adj_freq = scale_frequency_domain(data_out2_bits, freq)
    out1, data_out2, data_out2_bits = unknown3(available_bits, 16, adj_freq, data_out2_bits, bit_table1, bit_table2)
    compressed = to_bitstream(data_out2, data_out2_bits, data_out1_bits, data_out1, out1, 4)
    return compressed


def calc_number_of_bits_2(energy):
    if energy == 0:
        return -16
    bits = int(math.log(energy, 2))
    bits = max(0, bits - 15)
    energy >>= bits
    while energy < 0x8000 and bits > -16:
        energy *= 2
        bits -= 1
    if energy // 2 > 28962:
        bits += 1
    return bits


def unknown1(freq, some_number_of_bits):
    data_out1_bits = np.asarray([None] * 16)
    data_out1 = np.asarray([None] * 16)
    bits_per_freq_bin = np.asarray([None] * 14)

    for j in range(freq_bins):
        energy = int(np.sum(freq[j * 20 : j * 20 + 20] ** 2))
        bits_per_freq_bin[j] = 11 + calc_number_of_bits_2(energy) - some_number_of_bits * 2

    for i in range(freq_bins - 2, -1, -1):
        bits_per_freq_bin[i] = max(bits_per_freq_bin[i], bits_per_freq_bin[i + 1] - 11)

    bits_per_freq_bin = np.clip(bits_per_freq_bin, -15, 24)
    bits_per_freq_bin[0] = np.clip(bits_per_freq_bin[0], -6, 24)

    data_out1_bits[0] = 5
    data_out1[0] = bits_per_freq_bin[0] + 7
    for i in range(freq_bins - 1):
        sVar1 = max(-12, bits_per_freq_bin[i + 1] - bits_per_freq_bin[i]) + 12
        bits_per_freq_bin[i + 1] = bits_per_freq_bin[i] + sVar1 - 12
        data_out1_bits[i + 1] = SHORT_ARRAY_1000cea8[i + 1][sVar1]
        data_out1[i + 1] = SHORT_ARRAY_1000cea8_150[i + 1][sVar1]

    return bits_per_freq_bin, data_out1, data_out1_bits


def scale_frequency_domain(bits_per_freq_bin, freq):
    T1 = (bits_per_freq_bin - 39) // 2
    adj_freq = []
    for i in range(freq_bins):
        T = freq[i * 20 : i * 20 + 20]
        if T1[i] > 0:
            T = (T * 65536 + 0x8000) // (2 ** T1[i]) // 65536
        adj_freq = np.concatenate((adj_freq, T))
    adj_freq = np.concatenate((adj_freq, freq[len(adj_freq) :]))
    new_bits_per_freq_bin = np.where(T1 > 0, bits_per_freq_bin - T1 * 2, bits_per_freq_bin)
    return new_bits_per_freq_bin, adj_freq

def unknown3(available_bits, num16, adj_freq, new_bits_per_freq_bin, bit_table1, bit_table2):
    data_out2 = np.asarray([None] * freq_bins)

    bit_table1[bit_table2[: num16 // 2 - 1]] += 1
    data_out2_bits = [0] * freq_bins
    for i in range(freq_bins):
        if bit_table1[i] < 7:
            data_out2_bits[i], new_data = unknown4(bit_table1[i], new_bits_per_freq_bin[i], adj_freq[i * 20:i*20+20])
            data_out2[i] = new_data
    data_out2_bits = np.asarray(data_out2_bits)

    ret1_ = num16 // 2 - 1
    sVar6_ = np.sum(data_out2_bits)
    while sVar6_ < available_bits and ret1_ > 0:
        ret1_ -= 1
        idx = bit_table2[ret1_]
        bit_table1[idx] -= 1
        prv = data_out2_bits[idx]
        data_out2_bits[idx] = 0
        if bit_table1[idx] < 7:
            data_out2_bits[idx], new_data = unknown4(bit_table1[idx], new_bits_per_freq_bin[idx], adj_freq[idx * 20:idx*20+20])
            data_out2[idx] = new_data
        sVar6_ += data_out2_bits[idx] - prv

    while sVar6_ > available_bits and ret1_ < num16 - 1:
        idx = bit_table2[ret1_]
        bit_table1[idx] += 1
        prv = data_out2_bits[idx]
        data_out2_bits[idx] = 0
        if bit_table1[idx] < 7:
            data_out2_bits[idx], new_data = unknown4(bit_table1[idx], new_bits_per_freq_bin[idx], adj_freq[idx * 20:idx*20+20])
            data_out2[idx] = new_data
        sVar6_ += data_out2_bits[idx] - prv
        ret1_ += 1

    return ret1_, data_out2, data_out2_bits


def unknown4(bits, bits2, adj_freq):
    adj_freq_idx = 0
    sVar7 = (lookup_table2_80[bits] * lookup_table2_40[bits2] + 0x1000) // 32768
    sVar9 = lookup_table2_90[bits]
    sVar6 = lookup_table2_98[bits] + 1
    bitstream = BitStream(word_size=32)
    if sVar9 < 1:
        return None
    for j in range(sVar9):
        countbits = 0
        acc2 = 0
        sVar11_ = 0
        for i in range(lookup_table2_88[bits]):
            sVar8_ = (np.abs(adj_freq[adj_freq_idx]) * sVar7 + lookup_table2_a8[bits]) // 8192
            if sVar8_ != 0:
                countbits += 1
                acc2 *= 2
                if adj_freq[adj_freq_idx] > 0:
                    acc2 += 1
                sVar8_ = min(sVar8_, lookup_table2_98[bits])
            sVar11_ = (int(sVar11_ * sVar6 * 2) >> 1) + int(sVar8_)
            adj_freq_idx += 1
        val = (PTR_SHORT_ARRAY_1000f19c[bits][sVar11_] << countbits) + acc2
        bits_needed = PTR_SHORT_ARRAY_1000f180[bits][sVar11_] + countbits
        bitstream.push(val, bits_needed)
    return bitstream.bits, bitstream.acc


def to_bitstream(data_out2, data_out2_bits, data_out1_bits, data_out1, out1, number4):
    data_out1_bits[freq_bins] = number4
    data_out1[freq_bins] = out1

    bitstream = BitStream()
    for i in range(freq_bins + 1):
        bitstream.push(int(data_out1[i] & 0xFFFF), data_out1_bits[i])

    for i in range(freq_bins):
        width = int(data_out2_bits[i])
        if width == 0:
            continue
        bitstream.push(int(data_out2[i]), width)

    if bitstream.bits > total_bits:
        bitstream.acc >>= bitstream.bits - total_bits
        bitstream.bits = total_bits

    while bitstream.complete_words() * 16 < total_bits:
        width = 16 - bitstream.bits % 16
        bitstream.push((1 << width) - 1, width)

    return bitstream.tolist()


def unknown2(available_bits, num16, bits_per_freq_bin):
    if available_bits > 320:
        assert False

    correct_no_of_bits = find_correct_no_of_bits(bits_per_freq_bin, available_bits)

    adjusted_bits_per_freq_bin = unknown5(bits_per_freq_bin, correct_no_of_bits)

    bit_table1, bit_table2 = unknown6(adjusted_bits_per_freq_bin, bits_per_freq_bin, available_bits, num16, correct_no_of_bits)

    return bit_table1, bit_table2


def find_correct_no_of_bits(bits_per_freq_bin, available_bits):
    TT = np.asarray(lookup_table3)

    uVar1 = -32
    for j in [32, 16, 8, 4, 2, 1]:
        T = np.clip((uVar1 + j - bits_per_freq_bin) // 2, 0, 7)
        if np.sum(TT[T.astype(int)]) >= available_bits - 32:
            uVar1 += j

    return uVar1


def unknown5(bits_per_freq_bin, correct_no_of_bits):
    adjusted_bits_per_freq_bin = np.clip((correct_no_of_bits - bits_per_freq_bin) // 2, 0, 7)
    return adjusted_bits_per_freq_bin.tolist()

def unknown6(adjusted_bits_per_freq_bin, bits_per_freq_bin, available_bits, num16, correct_no_of_bits):
    LT3 = np.asarray(lookup_table3)
    total_bits = np.sum(LT3[adjusted_bits_per_freq_bin])
    tmp = [None] * 32

    total_bits2 = total_bits
    down = num16
    up = num16

    T1 = np.asarray(adjusted_bits_per_freq_bin[:freq_bins])
    T2 = np.asarray(adjusted_bits_per_freq_bin[:freq_bins])
    for j in range(num16 - 1):
        if total_bits2 + total_bits < available_bits * 2 + 1:
            only = np.where(T1 > 0)[0]
            idx = only[np.argmin(correct_no_of_bits - bits_per_freq_bin[only] - T1[only] * 2)]
            down -= 1
            T1[idx] -= 1
            tmp[down] = idx
            total_bits2 += lookup_table3[T1[idx]] - lookup_table3[T1[idx] + 1]
        else:
            only = np.flip(np.where(T2 < 7)[0])
            idx = only[np.argmax(correct_no_of_bits - bits_per_freq_bin[only] - T2[only] * 2)]
            tmp[up] = idx
            up += 1
            if T2[idx] < 7:
                T2[idx] += 1
                total_bits += lookup_table3[T2[idx]] - lookup_table3[T2[idx] - 1]

    bit_table2 = [None] * 15
    for i in range(num16 - 1):
        bit_table2[i] = tmp[down + i]

    bit_table1 = T1
    return bit_table1, bit_table2
