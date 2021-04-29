#ifndef __input_graph_h_
#define __input_graph_h_

#include <stdint.h>

#define NUM_NODES 100
#define NUM_EDGES 582

uint NODES[101] = {
0, 4, 14, 17, 51, 79, 107, 120, 128, 148, 157, 164, 183, 189, 194, 202, 207, 217, 224, 234, 239, 246, 252, 258, 267, 275, 280, 289, 292, 297, 303, 308, 311, 319, 323, 327, 332, 336, 340, 347, 351, 356, 359, 365, 369, 373, 377, 380, 387, 391, 396, 402, 407, 412, 416, 421, 425, 430, 433, 436, 439, 442, 449, 452, 459, 462, 465, 470, 473, 478, 481, 484, 488, 491, 495, 499, 502, 506, 511, 515, 518, 521, 525, 529, 532, 537, 540, 543, 546, 549, 552, 555, 558, 561, 564, 567, 570, 573, 576, 579, 582};

uint EDGES[582] = {
3, 4, 34, 69, 3, 4, 5, 7, 26, 32, 33, 40, 46, 73, 3, 55, 60, 0, 1, 2, 4, 5, 6, 8, 10, 12, 13, 14, 17, 20, 23, 26, 29, 31, 35, 37, 40, 41, 43, 44, 47, 50, 51, 61, 64, 65, 68, 70, 74, 94, 96, 0, 1, 3, 5, 6, 8, 9, 11, 14, 15, 18, 21, 22, 24, 25, 29, 33, 38, 41, 53, 54, 56, 62, 63, 70, 71, 77, 86, 1, 3, 4, 6, 7, 8, 9, 10, 11, 17, 18, 20, 25, 28, 30, 32, 36, 38, 45, 46, 74, 75, 76, 77, 85, 96, 98, 99, 3, 4, 5, 7, 13, 15, 16, 23, 31, 47, 54, 62, 95, 1, 5, 6, 21, 60, 71, 79, 90, 3, 4, 5, 9, 10, 12, 13, 14, 16, 19, 26, 27, 31, 44, 47, 49, 58, 64, 80, 83, 4, 5, 8, 15, 42, 57, 66, 73, 78, 3, 5, 8, 11, 30, 49, 82, 4, 5, 10, 12, 16, 17, 22, 24, 27, 45, 46, 51, 55, 56, 59, 63, 75, 77, 97, 3, 8, 11, 54, 81, 84, 3, 6, 8, 20, 48, 3, 4, 8, 18, 24, 48, 63, 97, 4, 6, 9, 51, 64, 6, 8, 11, 21, 22, 30, 32, 39, 50, 97, 3, 5, 11, 19, 53, 57, 81, 4, 5, 14, 19, 23, 25, 36, 39, 45, 99, 8, 17, 18, 35, 96, 3, 5, 13, 28, 66, 92, 99, 4, 7, 16, 42, 48, 91, 4, 11, 16, 56, 68, 88, 3, 6, 18, 34, 37, 42, 60, 61, 90, 4, 11, 14, 27, 38, 58, 65, 87, 4, 5, 18, 33, 87, 1, 3, 8, 28, 29, 39, 41, 61, 82, 8, 11, 24, 5, 20, 26, 35, 52, 3, 4, 26, 34, 43, 59, 5, 10, 16, 40, 88, 3, 6, 8, 1, 5, 16, 44, 50, 55, 58, 59, 1, 4, 25, 65, 0, 23, 29, 36, 3, 19, 28, 43, 83, 5, 18, 34, 37, 3, 23, 36, 67, 4, 5, 24, 72, 76, 80, 98, 16, 18, 26, 69, 1, 3, 30, 53, 66, 3, 4, 26, 9, 21, 23, 49, 72, 89, 3, 29, 35, 72, 3, 8, 32, 81, 5, 11, 18, 52, 1, 5, 11, 3, 6, 8, 52, 57, 85, 90, 13, 14, 21, 91, 8, 10, 42, 79, 88, 3, 16, 32, 62, 67, 76, 3, 11, 15, 78, 79, 28, 45, 47, 83, 84, 4, 17, 40, 69, 4, 6, 12, 70, 78, 2, 11, 32, 98, 4, 11, 22, 67, 92, 9, 17, 47, 8, 24, 32, 11, 29, 32, 2, 7, 23, 3, 23, 26, 74, 84, 92, 93, 4, 6, 50, 4, 11, 14, 75, 82, 85, 91, 3, 8, 15, 3, 24, 33, 9, 20, 40, 68, 73, 37, 50, 56, 3, 22, 66, 71, 95, 0, 39, 53, 3, 4, 54, 4, 7, 68, 86, 38, 42, 43, 1, 9, 66, 94, 3, 5, 61, 94, 5, 11, 63, 5, 38, 50, 80, 4, 5, 11, 93, 95, 9, 51, 54, 89, 7, 49, 51, 8, 38, 76, 12, 17, 44, 89, 10, 26, 63, 86, 8, 35, 52, 12, 52, 61, 87, 93, 5, 47, 63, 4, 71, 82, 24, 25, 84, 22, 30, 49, 42, 78, 81, 7, 23, 47, 21, 48, 63, 20, 56, 61, 61, 77, 84, 3, 73, 74, 6, 68, 77, 3, 5, 19, 11, 14, 16, 5, 38, 55, 5, 18, 20};

uint WEIGHTS[582] = {
7, 18, 4, 2, 9, 12, 15, 8, 9, 19, 17, 19, 12, 8, 13, 12, 7, 5, 11, 17, 8, 5, 9, 12, 12, 13, 10, 8, 2, 4, 9, 2, 14, 10, 16, 7, 13, 16, 1, 17, 2, 8, 19, 14, 18, 11, 1, 17, 13, 18, 9, 3, 7, 1, 3, 5, 5, 3, 11, 6, 10, 13, 10, 6, 6, 3, 11, 1, 17, 19, 13, 12, 14, 13, 13, 17, 3, 7, 5, 19, 16, 14, 1, 5, 5, 11, 7, 6, 6, 19, 2, 16, 10, 1, 6, 7, 14, 11, 15, 10, 18, 5, 5, 12, 15, 4, 2, 13, 2, 6, 1, 11, 7, 18, 18, 14, 5, 10, 10, 6, 1, 4, 11, 4, 2, 8, 12, 19, 16, 6, 11, 2, 9, 18, 15, 4, 1, 2, 6, 10, 12, 3, 7, 6, 4, 7, 8, 19, 2, 15, 17, 1, 10, 17, 10, 17, 17, 15, 15, 16, 4, 9, 16, 7, 6, 9, 9, 5, 4, 7, 14, 14, 16, 4, 2, 14, 8, 4, 7, 5, 14, 6, 15, 16, 4, 13, 19, 16, 3, 10, 2, 11, 14, 2, 10, 4, 16, 8, 15, 16, 4, 13, 12, 3, 3, 2, 12, 12, 4, 10, 6, 9, 17, 8, 2, 10, 12, 8, 5, 17, 4, 16, 17, 5, 9, 2, 4, 15, 18, 8, 2, 2, 19, 9, 8, 18, 5, 4, 8, 6, 18, 13, 8, 13, 7, 16, 2, 1, 14, 6, 12, 5, 12, 15, 15, 12, 17, 16, 6, 14, 5, 17, 15, 19, 2, 2, 18, 2, 6, 18, 13, 2, 18, 17, 7, 15, 18, 18, 15, 15, 16, 16, 17, 18, 10, 14, 7, 7, 15, 9, 6, 13, 17, 16, 1, 19, 12, 12, 10, 17, 18, 5, 13, 19, 7, 16, 17, 19, 14, 5, 19, 12, 5, 5, 8, 3, 2, 16, 4, 16, 9, 8, 10, 4, 2, 6, 13, 16, 11, 7, 12, 9, 2, 1, 6, 15, 19, 15, 12, 13, 9, 3, 6, 17, 1, 15, 16, 14, 5, 14, 1, 8, 10, 2, 17, 14, 4, 7, 18, 13, 17, 9, 15, 2, 4, 5, 9, 17, 2, 18, 2, 1, 5, 2, 8, 17, 6, 15, 2, 4, 4, 15, 5, 18, 19, 11, 8, 16, 15, 10, 8, 2, 11, 1, 9, 7, 13, 4, 8, 1, 16, 1, 3, 8, 15, 5, 13, 1, 3, 18, 13, 19, 2, 6, 15, 8, 14, 9, 15, 6, 6, 7, 15, 3, 2, 3, 16, 6, 19, 16, 2, 2, 5, 19, 14, 12, 17, 7, 18, 19, 10, 7, 7, 10, 14, 8, 8, 18, 7, 4, 16, 7, 14, 17, 18, 17, 14, 7, 9, 1, 10, 2, 10, 7, 12, 19, 6, 17, 18, 9, 6, 18, 16, 2, 5, 17, 13, 9, 2, 18, 18, 11, 14, 7, 11, 4, 10, 9, 6, 12, 2, 8, 13, 18, 4, 10, 9, 17, 10, 16, 18, 15, 6, 1, 14, 6, 13, 6, 10, 14, 13, 11, 19, 7, 3, 17, 19, 3, 10, 14, 6, 18, 17, 13, 9, 17, 10, 6, 2, 18, 6, 13, 1, 14, 18, 5, 17, 13, 18, 9, 1, 9, 9, 6, 12, 17, 17, 16, 16, 16, 5, 1, 1, 14, 19, 19, 19, 14, 16, 8, 13, 10, 14, 11, 6, 13, 2, 3, 7, 11, 11, 13, 11, 12, 16, 6, 9, 9, 4, 15, 9, 3};

#endif