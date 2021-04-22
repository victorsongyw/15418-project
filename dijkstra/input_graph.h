#ifndef __input_graph_h_
#define __input_graph_h_

#include <stdint.h>

#define NUM_NODES 100
#define NUM_EDGES 582

uint NODES[101] = {
0, 12, 26, 45, 68, 88, 112, 121, 127, 149, 161, 167, 172, 180, 191, 202, 207, 213, 220, 223, 230, 239, 242, 250, 259, 262, 268, 275, 282, 286, 294, 303, 312, 315, 320, 325, 330, 337, 345, 350, 356, 362, 367, 371, 376, 384, 387, 392, 397, 402, 405, 408, 411, 416, 419, 423, 426, 430, 434, 441, 446, 449, 454, 458, 463, 466, 470, 473, 476, 479, 482, 485, 490, 495, 498, 502, 505, 509, 512, 515, 519, 522, 527, 530, 533, 536, 539, 542, 546, 549, 552, 555, 558, 561, 564, 567, 570, 573, 576, 579, 582};

uint EDGES[582] = {
3, 4, 5, 8, 14, 31, 35, 61, 77, 79, 88, 91, 3, 4, 7, 9, 10, 13, 17, 19, 41, 50, 70, 79, 81, 87, 3, 4, 5, 7, 9, 12, 13, 15, 17, 19, 24, 40, 41, 43, 46, 47, 56, 71, 78, 0, 1, 2, 6, 7, 18, 21, 23, 24, 26, 27, 29, 30, 35, 36, 40, 44, 53, 66, 67, 76, 80, 84, 0, 1, 2, 5, 6, 8, 15, 16, 22, 24, 30, 32, 34, 42, 55, 58, 66, 68, 78, 92, 0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25, 28, 31, 33, 36, 37, 44, 45, 51, 53, 71, 77, 89, 96, 3, 4, 5, 23, 29, 46, 59, 62, 65, 1, 2, 3, 11, 37, 72, 0, 4, 5, 9, 10, 11, 12, 13, 14, 22, 25, 30, 31, 50, 55, 64, 69, 73, 74, 75, 80, 99, 1, 2, 8, 11, 17, 19, 29, 33, 45, 61, 67, 85, 1, 5, 8, 20, 23, 26, 7, 8, 9, 52, 75, 2, 5, 8, 18, 48, 69, 82, 87, 1, 2, 8, 15, 18, 20, 22, 28, 37, 81, 91, 0, 5, 8, 34, 49, 54, 68, 69, 70, 71, 90, 2, 4, 13, 16, 56, 4, 5, 15, 21, 63, 97, 1, 2, 9, 38, 80, 85, 90, 3, 12, 13, 1, 2, 9, 47, 77, 94, 97, 5, 10, 13, 21, 25, 26, 52, 95, 96, 3, 16, 20, 4, 8, 13, 27, 48, 54, 72, 93, 3, 6, 10, 32, 42, 60, 62, 65, 74, 2, 3, 4, 5, 8, 20, 27, 28, 94, 3, 10, 20, 33, 47, 63, 76, 3, 22, 25, 32, 34, 65, 95, 5, 13, 25, 86, 3, 6, 9, 35, 39, 40, 43, 45, 3, 4, 8, 38, 39, 49, 53, 54, 90, 0, 5, 8, 36, 56, 57, 79, 84, 88, 4, 23, 27, 5, 9, 26, 87, 93, 4, 14, 27, 43, 75, 0, 3, 29, 49, 52, 3, 5, 31, 38, 39, 41, 46, 5, 7, 13, 44, 59, 88, 98, 99, 17, 30, 36, 60, 68, 29, 30, 36, 66, 85, 95, 2, 3, 29, 42, 57, 63, 1, 2, 36, 50, 93, 4, 23, 40, 82, 2, 29, 34, 58, 89, 3, 5, 37, 48, 51, 59, 60, 83, 5, 9, 29, 2, 6, 36, 64, 73, 2, 19, 26, 51, 99, 12, 22, 44, 78, 94, 14, 30, 35, 1, 8, 41, 5, 44, 47, 11, 20, 35, 55, 89, 3, 5, 30, 14, 22, 30, 62, 4, 8, 52, 2, 15, 31, 57, 31, 40, 56, 58, 4, 43, 57, 70, 72, 82, 84, 6, 37, 44, 61, 64, 23, 38, 44, 0, 9, 59, 67, 73, 6, 23, 54, 86, 16, 26, 40, 83, 96, 8, 46, 59, 6, 23, 27, 83, 3, 4, 39, 3, 9, 61, 4, 14, 38, 8, 12, 14, 1, 14, 58, 2, 5, 14, 74, 92, 7, 22, 58, 76, 97, 8, 46, 61, 8, 23, 71, 91, 8, 11, 34, 3, 26, 72, 86, 0, 5, 19, 2, 4, 48, 0, 1, 31, 81, 3, 8, 17, 1, 13, 79, 92, 98, 12, 42, 58, 44, 63, 65, 3, 31, 58, 9, 17, 39, 28, 62, 76, 1, 12, 33, 98, 0, 31, 37, 5, 43, 52, 14, 17, 30, 0, 13, 74, 4, 71, 81, 22, 33, 41, 19, 25, 48, 20, 27, 39, 5, 20, 63, 16, 19, 72, 37, 81, 87, 8, 37, 47};

uint WEIGHTS[582] = {
9, 7, 7, 1, 2, 7, 3, 8, 8, 6, 2, 9, 1, 5, 5, 7, 8, 8, 9, 5, 1, 2, 3, 5, 3, 4, 3, 9, 1, 7, 4, 8, 5, 9, 8, 3, 4, 4, 9, 5, 9, 2, 5, 4, 3, 5, 7, 5, 7, 8, 4, 5, 6, 7, 8, 4, 5, 8, 4, 5, 7, 9, 6, 7, 7, 8, 2, 5, 7, 4, 9, 6, 5, 5, 1, 5, 3, 3, 6, 3, 9, 1, 4, 6, 4, 4, 8, 1, 3, 4, 4, 2, 4, 5, 7, 1, 5, 3, 2, 3, 8, 1, 7, 6, 6, 5, 8, 1, 3, 7, 1, 3, 1, 7, 9, 1, 3, 1, 6, 1, 2, 8, 5, 8, 3, 1, 4, 4, 8, 5, 7, 2, 9, 5, 6, 9, 4, 6, 5, 8, 1, 6, 5, 9, 7, 8, 5, 5, 2, 6, 5, 1, 5, 6, 9, 2, 7, 9, 2, 3, 3, 6, 8, 4, 5, 5, 4, 2, 5, 4, 6, 2, 7, 8, 8, 4, 6, 7, 4, 9, 2, 3, 1, 1, 6, 5, 3, 5, 5, 1, 4, 5, 3, 5, 8, 2, 6, 1, 1, 4, 6, 3, 1, 4, 7, 6, 7, 5, 8, 6, 9, 4, 5, 3, 2, 8, 4, 3, 2, 3, 4, 7, 4, 7, 4, 8, 5, 1, 6, 9, 4, 4, 6, 4, 6, 3, 3, 3, 2, 7, 9, 4, 6, 5, 8, 5, 3, 1, 1, 4, 8, 3, 2, 6, 3, 9, 7, 9, 7, 8, 2, 9, 1, 3, 9, 8, 8, 5, 9, 4, 3, 6, 4, 4, 7, 6, 1, 2, 8, 1, 6, 1, 5, 4, 3, 3, 4, 6, 5, 5, 7, 6, 9, 9, 1, 5, 8, 2, 4, 6, 4, 4, 6, 3, 6, 2, 1, 8, 5, 5, 1, 8, 1, 6, 9, 2, 8, 2, 1, 3, 8, 6, 2, 3, 4, 6, 8, 8, 3, 8, 6, 5, 7, 4, 1, 2, 2, 8, 6, 9, 3, 1, 5, 9, 3, 5, 7, 7, 5, 8, 3, 5, 4, 8, 2, 5, 4, 6, 1, 6, 2, 1, 8, 2, 8, 6, 6, 9, 6, 2, 9, 8, 5, 4, 9, 1, 4, 3, 3, 8, 6, 9, 8, 2, 3, 2, 2, 9, 7, 8, 5, 1, 9, 2, 3, 4, 2, 2, 4, 6, 8, 2, 9, 8, 2, 9, 3, 9, 2, 5, 3, 3, 8, 4, 6, 8, 9, 5, 7, 1, 8, 2, 8, 4, 3, 3, 8, 7, 2, 1, 5, 3, 1, 4, 4, 3, 9, 8, 9, 7, 3, 3, 1, 1, 2, 2, 5, 2, 2, 8, 8, 9, 9, 6, 8, 3, 5, 1, 6, 7, 7, 1, 7, 3, 6, 2, 7, 3, 3, 4, 1, 2, 5, 8, 9, 8, 1, 1, 5, 4, 5, 5, 2, 2, 1, 8, 7, 6, 3, 6, 4, 3, 2, 5, 2, 8, 9, 8, 2, 8, 5, 5, 2, 3, 2, 9, 3, 6, 1, 2, 1, 6, 5, 1, 2, 7, 2, 7, 1, 2, 7, 5, 8, 3, 6, 2, 8, 5, 3, 8, 9, 6, 2, 8, 5, 9, 5, 7, 3, 5, 1, 4, 1, 8, 1, 2, 6, 2, 1, 8, 7, 6, 6, 3, 8, 4, 2, 1, 3, 1, 4, 5, 3, 2, 5, 8, 3, 8, 5, 8, 2, 2, 7, 3, 1, 6, 7, 1, 3, 1, 7, 3};

#endif
