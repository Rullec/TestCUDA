#include "DihedralAnglesMachine.h"
namespace nDihedralAnglesMachine
{

_FLOAT CalcTheta(const tVector12 &xcur)
{
    Real copt2 = xcur(6);
    Real copt1 = xcur(4);
    Real copt5 = xcur(3);
    Real copt10 = xcur(7);
    Real copt4 = xcur(1);
    Real copt16 = xcur(9);
    Real copt9 = xcur(0);
    Real copt21 = xcur(10);
    Real copt32 = xcur(5);
    Real copt34 = xcur(8);
    Real copt28 = xcur(2);
    Real copt6 = -copt5;
    Real copt43 = xcur(11);
    Real copt23 = -copt1;
    Real copt36 = -copt32;
    Real copt68 = (copt9 * copt9);
    Real copt69 = (copt4 * copt4);
    Real copt70 = (copt28 * copt28);
    Real copt71 = -2 * copt5 * copt9;
    Real copt72 = (copt5 * copt5);
    Real copt73 = -2 * copt1 * copt4;
    Real copt74 = (copt1 * copt1);
    Real copt75 = -2 * copt28 * copt32;
    Real copt76 = (copt32 * copt32);
    Real copt77 = copt68 + copt69 + copt70 + copt71 + copt72 + copt73 + copt74 +
                  copt75 + copt76;
    Real copt78 = Sqrt(copt77);
    return ArcTan(
        ((copt10 + copt23) * copt28 - copt10 * copt32 + copt1 * copt34 +
         (copt32 - copt34) * copt4) *
                ((copt1 - copt21) * copt28 + copt21 * copt32 - copt1 * copt43 +
                 copt4 * (copt36 + copt43)) +
            (-(copt1 * copt2) + copt10 * copt5 + copt4 * (copt2 + copt6) +
             (copt1 - copt10) * copt9) *
                (copt1 * copt16 - copt21 * copt5 + copt4 * (-copt16 + copt5) +
                 (copt21 + copt23) * copt9) +
            (copt2 * copt32 - copt34 * copt5 + copt28 * (-copt2 + copt5) +
             (copt34 + copt36) * copt9) *
                (-(copt16 * copt32) + copt43 * copt5 +
                 copt28 * (copt16 + copt6) + (copt32 - copt43) * copt9),
        1. * copt78 *
            (copt10 * copt16 * copt32 - 1. * copt2 * copt21 * copt32 -
             1. * copt1 * copt16 * copt34 + copt1 * copt2 * copt43 +
             copt21 * copt34 * copt5 - 1. * copt10 * copt43 * copt5 +
             copt28 * (-1. * copt10 * copt16 + copt1 * (copt16 - 1. * copt2) +
                       copt2 * copt21 + (copt10 - 1. * copt21) * copt5) +
             copt4 * ((-1. * copt16 + copt2) * copt32 + copt16 * copt34 -
                      1. * copt2 * copt43 + (-1. * copt34 + copt43) * copt5) -
             1. * copt10 * copt32 * copt9 + copt21 * copt32 * copt9 +
             copt1 * copt34 * copt9 - 1. * copt21 * copt34 * copt9 -
             1. * copt1 * copt43 * copt9 + copt10 * copt43 * copt9));
}

tVector12 CalcdThetadx(const tVector12 &xcur)
{
    Real copt1 = xcur(0);
    Real copt2 = (copt1 * copt1);
    Real copt3 = xcur(1);
    Real copt4 = (copt3 * copt3);
    Real copt5 = xcur(2);
    Real copt6 = (copt5 * copt5);
    Real copt7 = xcur(3);
    Real copt8 = -2 * copt1 * copt7;
    Real copt9 = (copt7 * copt7);
    Real copt10 = xcur(4);
    Real copt11 = -2 * copt10 * copt3;
    Real copt12 = (copt10 * copt10);
    Real copt13 = xcur(5);
    Real copt14 = -2 * copt13 * copt5;
    Real copt15 = (copt13 * copt13);
    Real copt16 = copt11 + copt12 + copt14 + copt15 + copt2 + copt4 + copt6 +
                  copt8 + copt9;
    Real copt17 = Sqrt(copt16);
    Real copt18 = 1 / copt17;
    Real copt19 = xcur(6);
    Real copt20 = -(copt10 * copt19);
    Real copt21 = -copt7;
    Real copt22 = copt19 + copt21;
    Real copt23 = copt22 * copt3;
    Real copt24 = xcur(7);
    Real copt25 = -copt24;
    Real copt26 = copt10 + copt25;
    Real copt27 = copt1 * copt26;
    Real copt28 = copt24 * copt7;
    Real copt29 = copt20 + copt23 + copt27 + copt28;
    Real copt30 = xcur(9);
    Real copt31 = -copt30;
    Real copt32 = copt31 + copt7;
    Real copt33 = copt3 * copt32;
    Real copt34 = copt10 * copt30;
    Real copt35 = xcur(10);
    Real copt36 = -(copt35 * copt7);
    Real copt37 = -copt10;
    Real copt38 = copt35 + copt37;
    Real copt39 = copt1 * copt38;
    Real copt40 = copt33 + copt34 + copt36 + copt39;
    Real copt41 = copt29 * copt40;
    Real copt42 = -copt19;
    Real copt43 = copt42 + copt7;
    Real copt44 = copt43 * copt5;
    Real copt45 = copt13 * copt19;
    Real copt46 = xcur(8);
    Real copt47 = -(copt46 * copt7);
    Real copt48 = -copt13;
    Real copt49 = copt46 + copt48;
    Real copt50 = copt1 * copt49;
    Real copt51 = copt44 + copt45 + copt47 + copt50;
    Real copt52 = -(copt13 * copt30);
    Real copt53 = copt21 + copt30;
    Real copt54 = copt5 * copt53;
    Real copt55 = xcur(11);
    Real copt56 = -copt55;
    Real copt57 = copt13 + copt56;
    Real copt58 = copt1 * copt57;
    Real copt59 = copt55 * copt7;
    Real copt60 = copt52 + copt54 + copt58 + copt59;
    Real copt61 = copt51 * copt60;
    Real copt62 = -(copt13 * copt24);
    Real copt63 = copt24 + copt37;
    Real copt64 = copt5 * copt63;
    Real copt65 = -copt46;
    Real copt66 = copt13 + copt65;
    Real copt67 = copt3 * copt66;
    Real copt68 = copt10 * copt46;
    Real copt69 = copt62 + copt64 + copt67 + copt68;
    Real copt70 = -copt35;
    Real copt71 = copt10 + copt70;
    Real copt72 = copt5 * copt71;
    Real copt73 = copt13 * copt35;
    Real copt74 = -(copt10 * copt55);
    Real copt75 = copt48 + copt55;
    Real copt76 = copt3 * copt75;
    Real copt77 = copt72 + copt73 + copt74 + copt76;
    Real copt78 = copt69 * copt77;
    Real copt79 = copt41 + copt61 + copt78;
    Real copt80 = (copt79 * copt79);
    Real copt81 = -1. * copt1 * copt13 * copt24;
    Real copt82 = copt1 * copt10 * copt46;
    Real copt83 = copt13 * copt24 * copt30;
    Real copt84 = -1. * copt10 * copt30 * copt46;
    Real copt85 = copt1 * copt13 * copt35;
    Real copt86 = -1. * copt13 * copt19 * copt35;
    Real copt87 = -1. * copt1 * copt35 * copt46;
    Real copt88 = copt35 * copt46 * copt7;
    Real copt89 = -1. * copt24 * copt30;
    Real copt90 = -1. * copt19;
    Real copt91 = copt30 + copt90;
    Real copt92 = copt10 * copt91;
    Real copt93 = -1. * copt35;
    Real copt94 = copt24 + copt93;
    Real copt95 = copt7 * copt94;
    Real copt96 = copt19 * copt35;
    Real copt97 = copt89 + copt92 + copt95 + copt96;
    Real copt98 = copt5 * copt97;
    Real copt99 = -1. * copt1 * copt10 * copt55;
    Real copt100 = copt10 * copt19 * copt55;
    Real copt101 = copt1 * copt24 * copt55;
    Real copt102 = -1. * copt24 * copt55 * copt7;
    Real copt103 = -1. * copt30;
    Real copt104 = copt103 + copt19;
    Real copt105 = copt104 * copt13;
    Real copt106 = copt30 * copt46;
    Real copt107 = -1. * copt19 * copt55;
    Real copt108 = -1. * copt46;
    Real copt109 = copt108 + copt55;
    Real copt110 = copt109 * copt7;
    Real copt111 = copt105 + copt106 + copt107 + copt110;
    Real copt112 = copt111 * copt3;
    Real copt113 = copt100 + copt101 + copt102 + copt112 + copt81 + copt82 +
                   copt83 + copt84 + copt85 + copt86 + copt87 + copt88 +
                   copt98 + copt99;
    Real copt114 = (copt113 * copt113);
    Real copt115 = 1. * copt114 * copt16;
    Real copt116 = copt115 + copt80;
    Real copt117 = 1 / copt116;
    Real copt177 = -copt5;
    Real copt174 = -copt3;
    Real copt197 = -copt1;
    Real copt132 = -1. * copt55;
    Real copt151 = copt3 + copt37;
    Real copt138 = copt1 + copt21;
    Real copt129 = -1. * copt24;
    Real copt286 = copt10 + copt174;
    Real copt271 = copt48 + copt5;
    Real copt278 = -1. * copt7;
    Real copt256 = copt13 + copt177;
    Real copt269 = copt197 + copt7;
    tVector12 out;
    out(0) =
        copt117 * copt18 *
        (-1. * copt113 * copt16 *
             (copt29 * copt38 + copt26 * copt40 + copt51 * copt57 +
              copt49 * copt60) +
         copt79 *
             (1. * copt113 * copt138 +
              1. *
                  (copt13 * (copt129 + copt35) - 1. * copt35 * copt46 +
                   copt10 * (copt132 + copt46) + copt24 * copt55) *
                  (copt12 + copt15 + copt2 - 2. * copt10 * copt3 + copt4 -
                   2. * copt13 * copt5 + copt6 - 2. * copt1 * copt7 + copt9)));
    out(1) = copt117 * copt18 *
             (-1. * copt113 * copt16 *
                  (copt29 * copt32 + copt22 * copt40 + copt69 * copt75 +
                   copt66 * copt77) +
              (1. * copt113 * copt151 + 1. * copt111 * copt16) * copt79);
    out(2) = copt117 *
             (-1. * copt113 * copt17 *
                  (copt51 * copt53 + copt43 * copt60 + copt69 * copt71 +
                   copt63 * copt77) +
              copt79 * (0.5 * copt113 * copt18 * (-2 * copt13 + 2 * copt5) +
                        1. * copt17 * copt97));
    out(3) = -1. * copt113 * copt117 * copt17 *
                 ((copt174 + copt24) * copt40 + copt51 * (copt177 + copt55) +
                  copt60 * (copt5 + copt65) + copt29 * (copt3 + copt70)) +
             copt117 * copt79 *
                 (0.5 * copt113 * copt18 * (-2 * copt1 + 2 * copt7) +
                  1. * copt17 *
                      (copt109 * copt3 + copt35 * copt46 -
                       1. * copt24 * copt55 + copt5 * copt94));
    out(4) = -1. * copt113 * copt117 * copt17 *
                 (copt29 * (copt197 + copt30) + copt40 * (copt1 + copt42) +
                  (copt5 + copt56) * copt69 + (copt177 + copt46) * copt77) +
             copt117 * copt79 *
                 (0.5 * copt113 * copt18 * (2 * copt10 - 2 * copt3) +
                  1. * copt17 *
                      (copt1 * copt46 - 1. * copt30 * copt46 -
                       1. * copt1 * copt55 + copt19 * copt55 + copt5 * copt91));
    out(5) =
        -1. * copt113 * copt117 * copt17 *
            ((copt1 + copt31) * copt51 + (copt19 + copt197) * copt60 +
             (copt174 + copt35) * copt69 + (copt25 + copt3) * copt77) +
        copt117 *
            (1. * copt17 *
                 (-1. * copt1 * copt24 + copt104 * copt3 + copt24 * copt30 +
                  copt1 * copt35 - 1. * copt19 * copt35) +
             0.5 * copt113 * copt18 * (2 * copt13 - 2 * copt5)) *
            copt79;
    out(6) = -1. * copt113 * copt117 * copt17 *
                 (copt151 * copt40 + copt256 * copt60) +
             1. * copt117 * copt17 *
                 ((copt13 + copt132) * copt3 - 1. * copt13 * copt35 +
                  (-1. * copt10 + copt35) * copt5 + copt10 * copt55) *
                 copt79;
    out(7) = -1. * copt113 * copt117 * copt17 *
                 (copt269 * copt40 + copt271 * copt77) +
             1. * copt117 * copt17 *
                 (-1. * copt1 * copt13 + copt13 * copt30 + copt1 * copt55 -
                  1. * copt55 * copt7 + copt5 * (copt103 + copt7)) *
                 copt79;
    out(8) = -1. * copt113 * copt117 * copt17 *
                 (copt138 * copt60 + copt286 * copt77) +
             1. * copt117 * copt17 *
                 (copt1 * copt10 - 1. * copt10 * copt30 +
                  copt3 * (copt278 + copt30) - 1. * copt1 * copt35 +
                  copt35 * copt7) *
                 copt79;
    out(9) =
        -1. * copt113 * copt117 * copt17 *
            (copt286 * copt29 + copt271 * copt51) +
        1. * copt117 * copt17 *
            (copt13 * copt24 - 1. * copt10 * copt46 +
             copt3 * (-1. * copt13 + copt46) + (copt10 + copt129) * copt5) *
            copt79;
    out(10) = -1. * copt113 * copt117 * copt17 *
                  (copt138 * copt29 + copt256 * copt69) +
              1. * copt117 * copt17 *
                  (copt1 * copt13 - 1. * copt13 * copt19 - 1. * copt1 * copt46 +
                   (copt19 + copt278) * copt5 + copt46 * copt7) *
                  copt79;
    out(11) = -1. * copt113 * copt117 * copt17 *
                  (copt269 * copt51 + copt151 * copt69) +
              1. * copt117 * copt17 * copt79 *
                  (-1. * copt1 * copt10 + copt10 * copt19 + copt1 * copt24 -
                   1. * copt24 * copt7 + copt3 * (copt7 + copt90));
    return out;
}

tMatrix12 CalcdTheta2dx2(const tVector12 &xcur)
{
    Real copt1 = xcur(0);
    Real copt2 = (copt1 * copt1);
    Real copt3 = xcur(1);
    Real copt4 = (copt3 * copt3);
    Real copt5 = xcur(2);
    Real copt6 = (copt5 * copt5);
    Real copt7 = xcur(3);
    Real copt8 = -2 * copt1 * copt7;
    Real copt9 = (copt7 * copt7);
    Real copt10 = xcur(4);
    Real copt11 = -2 * copt10 * copt3;
    Real copt12 = (copt10 * copt10);
    Real copt13 = xcur(5);
    Real copt14 = -2 * copt13 * copt5;
    Real copt15 = (copt13 * copt13);
    Real copt16 = copt11 + copt12 + copt14 + copt15 + copt2 + copt4 + copt6 +
                  copt8 + copt9;
    Real copt17 = Sqrt(copt16);
    Real copt18 = 1 / copt17;
    Real copt19 = xcur(6);
    Real copt20 = -(copt10 * copt19);
    Real copt21 = -copt7;
    Real copt22 = copt19 + copt21;
    Real copt23 = copt22 * copt3;
    Real copt24 = xcur(7);
    Real copt25 = -copt24;
    Real copt26 = copt10 + copt25;
    Real copt27 = copt1 * copt26;
    Real copt28 = copt24 * copt7;
    Real copt29 = copt20 + copt23 + copt27 + copt28;
    Real copt30 = xcur(9);
    Real copt31 = -copt30;
    Real copt32 = copt31 + copt7;
    Real copt33 = copt3 * copt32;
    Real copt34 = copt10 * copt30;
    Real copt35 = xcur(10);
    Real copt36 = -(copt35 * copt7);
    Real copt37 = -copt10;
    Real copt38 = copt35 + copt37;
    Real copt39 = copt1 * copt38;
    Real copt40 = copt33 + copt34 + copt36 + copt39;
    Real copt41 = copt29 * copt40;
    Real copt42 = -copt19;
    Real copt43 = copt42 + copt7;
    Real copt44 = copt43 * copt5;
    Real copt45 = copt13 * copt19;
    Real copt46 = xcur(8);
    Real copt47 = -(copt46 * copt7);
    Real copt48 = -copt13;
    Real copt49 = copt46 + copt48;
    Real copt50 = copt1 * copt49;
    Real copt51 = copt44 + copt45 + copt47 + copt50;
    Real copt52 = -(copt13 * copt30);
    Real copt53 = copt21 + copt30;
    Real copt54 = copt5 * copt53;
    Real copt55 = xcur(11);
    Real copt56 = -copt55;
    Real copt57 = copt13 + copt56;
    Real copt58 = copt1 * copt57;
    Real copt59 = copt55 * copt7;
    Real copt60 = copt52 + copt54 + copt58 + copt59;
    Real copt61 = copt51 * copt60;
    Real copt62 = -(copt13 * copt24);
    Real copt63 = copt24 + copt37;
    Real copt64 = copt5 * copt63;
    Real copt65 = -copt46;
    Real copt66 = copt13 + copt65;
    Real copt67 = copt3 * copt66;
    Real copt68 = copt10 * copt46;
    Real copt69 = copt62 + copt64 + copt67 + copt68;
    Real copt70 = -copt35;
    Real copt71 = copt10 + copt70;
    Real copt72 = copt5 * copt71;
    Real copt73 = copt13 * copt35;
    Real copt74 = -(copt10 * copt55);
    Real copt75 = copt48 + copt55;
    Real copt76 = copt3 * copt75;
    Real copt77 = copt72 + copt73 + copt74 + copt76;
    Real copt78 = copt69 * copt77;
    Real copt79 = copt41 + copt61 + copt78;
    Real copt80 = (copt79 * copt79);
    Real copt81 = -1. * copt1 * copt13 * copt24;
    Real copt82 = copt1 * copt10 * copt46;
    Real copt83 = copt13 * copt24 * copt30;
    Real copt84 = -1. * copt10 * copt30 * copt46;
    Real copt85 = copt1 * copt13 * copt35;
    Real copt86 = -1. * copt13 * copt19 * copt35;
    Real copt87 = -1. * copt1 * copt35 * copt46;
    Real copt88 = copt35 * copt46 * copt7;
    Real copt89 = -1. * copt24 * copt30;
    Real copt90 = -1. * copt19;
    Real copt91 = copt30 + copt90;
    Real copt92 = copt10 * copt91;
    Real copt93 = -1. * copt35;
    Real copt94 = copt24 + copt93;
    Real copt95 = copt7 * copt94;
    Real copt96 = copt19 * copt35;
    Real copt97 = copt89 + copt92 + copt95 + copt96;
    Real copt98 = copt5 * copt97;
    Real copt99 = -1. * copt1 * copt10 * copt55;
    Real copt100 = copt10 * copt19 * copt55;
    Real copt101 = copt1 * copt24 * copt55;
    Real copt102 = -1. * copt24 * copt55 * copt7;
    Real copt103 = -1. * copt30;
    Real copt104 = copt103 + copt19;
    Real copt105 = copt104 * copt13;
    Real copt106 = copt30 * copt46;
    Real copt107 = -1. * copt19 * copt55;
    Real copt108 = -1. * copt46;
    Real copt109 = copt108 + copt55;
    Real copt110 = copt109 * copt7;
    Real copt111 = copt105 + copt106 + copt107 + copt110;
    Real copt112 = copt111 * copt3;
    Real copt113 = copt100 + copt101 + copt102 + copt112 + copt81 + copt82 +
                   copt83 + copt84 + copt85 + copt86 + copt87 + copt88 +
                   copt98 + copt99;
    Real copt114 = (copt113 * copt113);
    Real copt115 = 1. * copt114 * copt16;
    Real copt116 = copt115 + copt80;
    Real copt117 = 1 / copt116;
    Real copt127 = copt29 * copt38;
    Real copt128 = copt26 * copt40;
    Real copt129 = copt51 * copt57;
    Real copt130 = copt49 * copt60;
    Real copt131 = copt127 + copt128 + copt129 + copt130;
    Real copt118 = -1. * copt35 * copt46;
    Real copt119 = -1. * copt24;
    Real copt120 = copt119 + copt35;
    Real copt121 = copt120 * copt13;
    Real copt122 = -1. * copt55;
    Real copt123 = copt122 + copt46;
    Real copt124 = copt10 * copt123;
    Real copt125 = copt24 * copt55;
    Real copt126 = copt118 + copt121 + copt124 + copt125;
    Real copt168 = copt1 + copt21;
    Real copt184 = (copt116 * copt116);
    Real copt185 = 1 / copt184;
    Real copt170 = -2. * copt1 * copt7;
    Real copt171 = -2. * copt10 * copt3;
    Real copt172 = -2. * copt13 * copt5;
    Real copt173 = copt12 + copt15 + copt170 + copt171 + copt172 + copt2 +
                   copt4 + copt6 + copt9;
    Real copt174 = 1. * copt126 * copt173;
    Real copt175 = 1. * copt113 * copt168;
    Real copt176 = copt174 + copt175;
    Real copt193 = copt16 * copt17;
    Real copt194 = 1 / copt193;
    Real copt186 = -1. * copt113 * copt131 * copt16;
    Real copt187 = copt176 * copt79;
    Real copt188 = copt186 + copt187;
    Real copt210 = copt29 * copt32;
    Real copt211 = copt22 * copt40;
    Real copt212 = copt69 * copt75;
    Real copt213 = copt66 * copt77;
    Real copt214 = copt210 + copt211 + copt212 + copt213;
    Real copt202 = copt3 + copt37;
    Real copt241 = copt51 * copt53;
    Real copt242 = copt69 * copt71;
    Real copt243 = copt43 * copt60;
    Real copt244 = copt63 * copt77;
    Real copt245 = copt241 + copt242 + copt243 + copt244;
    Real copt239 = copt48 + copt5;
    Real copt304 = copt3 + copt70;
    Real copt310 = copt5 + copt65;
    Real copt318 = copt29 * copt304;
    Real copt319 = -copt3;
    Real copt320 = copt24 + copt319;
    Real copt321 = copt320 * copt40;
    Real copt322 = -copt5;
    Real copt323 = copt322 + copt55;
    Real copt324 = copt323 * copt51;
    Real copt325 = copt310 * copt60;
    Real copt326 = copt318 + copt321 + copt324 + copt325;
    Real copt259 = copt5 * copt94;
    Real copt260 = copt35 * copt46;
    Real copt261 = -1. * copt24 * copt55;
    Real copt262 = copt109 * copt3;
    Real copt263 = copt259 + copt260 + copt261 + copt262;
    Real copt314 = -2 * copt1;
    Real copt315 = 2 * copt7;
    Real copt316 = copt314 + copt315;
    Real copt296 = -1. * copt30 * copt46;
    Real copt337 = copt5 * copt91;
    Real copt338 = copt1 * copt123;
    Real copt339 = copt19 * copt55;
    Real copt340 = copt296 + copt337 + copt338 + copt339;
    Real copt345 = 2 * copt10;
    Real copt308 = copt5 + copt56;
    Real copt367 = -copt1;
    Real copt368 = copt30 + copt367;
    Real copt369 = copt29 * copt368;
    Real copt370 = copt1 + copt42;
    Real copt371 = copt370 * copt40;
    Real copt372 = copt308 * copt69;
    Real copt373 = copt322 + copt46;
    Real copt374 = copt373 * copt77;
    Real copt375 = copt369 + copt371 + copt372 + copt374;
    Real copt364 = -2 * copt3;
    Real copt365 = copt345 + copt364;
    Real copt352 = copt24 * copt30;
    Real copt284 = -1. * copt19 * copt35;
    Real copt386 = -1. * copt1 * copt24;
    Real copt387 = copt104 * copt3;
    Real copt388 = copt1 * copt35;
    Real copt389 = copt284 + copt352 + copt386 + copt387 + copt388;
    Real copt394 = 2 * copt13;
    Real copt399 = copt1 + copt31;
    Real copt404 = copt19 + copt367;
    Real copt306 = copt25 + copt3;
    Real copt412 = copt399 * copt51;
    Real copt413 = copt319 + copt35;
    Real copt414 = copt413 * copt69;
    Real copt415 = copt404 * copt60;
    Real copt416 = copt306 * copt77;
    Real copt417 = copt412 + copt414 + copt415 + copt416;
    Real copt409 = -2 * copt5;
    Real copt410 = copt394 + copt409;
    Real copt204 = -1. * copt10;
    Real copt428 = -1. * copt13 * copt35;
    Real copt429 = copt204 + copt35;
    Real copt430 = copt429 * copt5;
    Real copt431 = copt122 + copt13;
    Real copt432 = copt3 * copt431;
    Real copt433 = copt10 * copt55;
    Real copt434 = copt428 + copt430 + copt432 + copt433;
    Real copt438 = copt13 + copt322;
    Real copt442 = copt202 * copt40;
    Real copt443 = copt438 * copt60;
    Real copt444 = copt442 + copt443;
    Real copt401 = copt13 * copt30;
    Real copt230 = -1. * copt13;
    Real copt453 = -1. * copt1 * copt13;
    Real copt454 = copt103 + copt7;
    Real copt455 = copt454 * copt5;
    Real copt456 = copt1 * copt55;
    Real copt457 = -1. * copt55 * copt7;
    Real copt458 = copt401 + copt453 + copt455 + copt456 + copt457;
    Real copt361 = copt35 * copt7;
    Real copt468 = copt367 + copt7;
    Real copt472 = copt40 * copt468;
    Real copt473 = copt239 * copt77;
    Real copt474 = copt472 + copt473;
    Real copt282 = -1. * copt1 * copt35;
    Real copt483 = copt1 * copt10;
    Real copt484 = -1. * copt10 * copt30;
    Real copt485 = -1. * copt7;
    Real copt486 = copt30 + copt485;
    Real copt487 = copt3 * copt486;
    Real copt488 = copt282 + copt361 + copt483 + copt484 + copt487;
    Real copt498 = copt168 * copt60;
    Real copt499 = copt10 + copt319;
    Real copt500 = copt499 * copt77;
    Real copt501 = copt498 + copt500;
    Real copt510 = copt10 + copt119;
    Real copt511 = copt5 * copt510;
    Real copt512 = copt13 * copt24;
    Real copt513 = -1. * copt10 * copt46;
    Real copt514 = copt230 + copt46;
    Real copt515 = copt3 * copt514;
    Real copt516 = copt511 + copt512 + copt513 + copt515;
    Real copt523 = copt29 * copt499;
    Real copt524 = copt239 * copt51;
    Real copt525 = copt523 + copt524;
    Real copt292 = -1. * copt13 * copt19;
    Real copt293 = -1. * copt1 * copt46;
    Real copt534 = copt1 * copt13;
    Real copt535 = copt19 + copt485;
    Real copt536 = copt5 * copt535;
    Real copt537 = copt46 * copt7;
    Real copt538 = copt292 + copt293 + copt534 + copt536 + copt537;
    Real copt548 = copt168 * copt29;
    Real copt549 = copt438 * copt69;
    Real copt550 = copt548 + copt549;
    Real copt559 = -1. * copt1 * copt10;
    Real copt560 = copt7 + copt90;
    Real copt561 = copt3 * copt560;
    Real copt562 = copt10 * copt19;
    Real copt563 = copt1 * copt24;
    Real copt564 = -1. * copt24 * copt7;
    Real copt565 = copt559 + copt561 + copt562 + copt563 + copt564;
    Real copt578 = copt202 * copt69;
    Real copt579 = copt468 * copt51;
    Real copt580 = copt578 + copt579;
    Real copt198 = copt26 * copt32;
    Real copt190 = 2 * copt1;
    Real copt191 = -2 * copt7;
    Real copt192 = copt190 + copt191;
    Real copt180 = 2 * copt131 * copt79;
    Real copt181 = 2. * copt113 * copt126 * copt16;
    Real copt182 = 2. * copt114 * copt168;
    Real copt183 = copt180 + copt181 + copt182;
    Real copt598 = 1. * copt111 * copt16;
    Real copt599 = 1. * copt113 * copt202;
    Real copt600 = copt598 + copt599;
    Real copt604 = -1. * copt113 * copt16 * copt214;
    Real copt605 = copt600 * copt79;
    Real copt606 = copt604 + copt605;
    Real copt223 = 2 * copt3;
    Real copt224 = -2 * copt10;
    Real copt225 = copt223 + copt224;
    Real copt218 = 2 * copt214 * copt79;
    Real copt219 = 2. * copt111 * copt113 * copt16;
    Real copt220 = 2. * copt114 * copt202;
    Real copt221 = copt218 + copt219 + copt220;
    Real copt254 = 2 * copt5;
    Real copt255 = -2 * copt13;
    Real copt256 = copt254 + copt255;
    Real copt249 = 2 * copt245 * copt79;
    Real copt250 = 2. * copt113 * copt16 * copt97;
    Real copt251 = 2. * copt114 * copt239;
    Real copt252 = copt249 + copt250 + copt251;
    Real copt465 = -(copt3 * copt32);
    Real copt466 = -(copt10 * copt30);
    Real copt467 = -(copt1 * copt38);
    Real copt330 = 2 * copt326 * copt79;
    Real copt331 = 2. * copt113 * copt16 * copt263;
    Real copt332 = 1. * copt114 * copt316;
    Real copt333 = copt330 + copt331 + copt332;
    Real copt309 = copt308 * copt66;
    Real copt379 = 2 * copt375 * copt79;
    Real copt380 = 2. * copt113 * copt16 * copt340;
    Real copt381 = 1. * copt114 * copt365;
    Real copt382 = copt379 + copt380 + copt381;
    Real copt421 = 2 * copt417 * copt79;
    Real copt422 = 2. * copt113 * copt16 * copt389;
    Real copt423 = 1. * copt114 * copt410;
    Real copt424 = copt421 + copt422 + copt423;
    Real copt448 = 2 * copt444 * copt79;
    Real copt449 = 2. * copt113 * copt16 * copt434;
    Real copt450 = copt448 + copt449;
    Real copt478 = 2 * copt474 * copt79;
    Real copt479 = 2. * copt113 * copt16 * copt458;
    Real copt480 = copt478 + copt479;
    Real copt505 = 2 * copt501 * copt79;
    Real copt506 = 2. * copt113 * copt16 * copt488;
    Real copt507 = copt505 + copt506;
    Real copt529 = 2 * copt525 * copt79;
    Real copt530 = 2. * copt113 * copt16 * copt516;
    Real copt531 = copt529 + copt530;
    Real copt554 = 2 * copt550 * copt79;
    Real copt555 = 2. * copt113 * copt16 * copt538;
    Real copt556 = copt554 + copt555;
    Real copt584 = 2 * copt580 * copt79;
    Real copt585 = 2. * copt113 * copt16 * copt565;
    Real copt586 = copt584 + copt585;
    Real copt236 = copt43 * copt57;
    Real copt787 = 1. * copt16 * copt97;
    Real copt788 = 1. * copt113 * copt239;
    Real copt789 = copt787 + copt788;
    Real copt793 = -1. * copt113 * copt16 * copt245;
    Real copt794 = copt789 * copt79;
    Real copt795 = copt793 + copt794;
    Real copt628 = copt66 * copt71;
    Real copt629 = copt63 * copt75;
    Real copt630 = copt628 + copt629;
    Real copt631 = -1. * copt113 * copt16 * copt630;
    Real copt618 = 1. * copt113;
    Real copt572 = -(copt43 * copt5);
    Real copt573 = -(copt13 * copt19);
    Real copt574 = -(copt1 * copt49);
    Real copt727 = -(copt5 * copt71);
    Real copt728 = -(copt13 * copt35);
    Real copt729 = -(copt3 * copt75);
    Real copt307 = copt306 * copt71;
    Real copt669 = -1. * copt113;
    Real copt402 = -(copt5 * copt53);
    Real copt403 = -(copt1 * copt57);
    Real copt406 = -(copt55 * copt7);
    Real copt679 = -(copt5 * copt63);
    Real copt680 = -(copt3 * copt66);
    Real copt681 = -(copt10 * copt46);
    Real copt305 = copt26 * copt304;
    Real copt311 = copt310 * copt57;
    Real copt962 = 1. * copt16 * copt263;
    Real copt963 = 0.5 * copt113 * copt316;
    Real copt964 = copt962 + copt963;
    Real copt968 = -1. * copt113 * copt16 * copt326;
    Real copt969 = copt79 * copt964;
    Real copt970 = copt968 + copt969;
    Real copt644 = copt32 * copt320;
    Real copt645 = copt22 * copt304;
    Real copt646 = copt20 + copt23 + copt27 + copt28 + copt361 + copt465 +
                   copt466 + copt467 + copt644 + copt645;
    Real copt647 = -1. * copt113 * copt16 * copt646;
    Real copt649 = 1. * copt109 * copt16;
    Real copt828 = 1. * copt16 * copt94;
    Real copt833 = copt310 * copt53;
    Real copt834 = copt323 * copt43;
    Real copt835 = copt52 + copt537 + copt54 + copt572 + copt573 + copt574 +
                   copt58 + copt59 + copt833 + copt834;
    Real copt836 = -1. * copt113 * copt16 * copt835;
    Real copt743 = -(copt22 * copt3);
    Real copt745 = -(copt1 * copt26);
    Real copt746 = -(copt24 * copt7);
    Real copt350 = -2. * copt10 * copt19;
    Real copt351 = -2. * copt10 * copt30;
    Real copt353 = -2. * copt7;
    Real copt354 = copt19 + copt30 + copt353;
    Real copt355 = copt3 * copt354;
    Real copt356 = 4. * copt10;
    Real copt357 = -2. * copt24;
    Real copt358 = -2. * copt35;
    Real copt359 = copt356 + copt357 + copt358;
    Real copt360 = copt1 * copt359;
    Real copt362 = copt28 + copt350 + copt351 + copt352 + copt355 + copt360 +
                   copt361 + copt96;
    Real copt363 = 1. * copt113 * copt173 * copt362;
    Real copt1132 = 1. * copt16 * copt340;
    Real copt1133 = 0.5 * copt113 * copt365;
    Real copt1134 = copt1132 + copt1133;
    Real copt1138 = -1. * copt113 * copt16 * copt375;
    Real copt1139 = copt1134 * copt79;
    Real copt1140 = copt1138 + copt1139;
    Real copt661 = copt32 * copt370;
    Real copt662 = copt22 * copt368;
    Real copt663 = copt373 * copt75;
    Real copt664 = copt309 + copt661 + copt662 + copt663;
    Real copt665 = -1. * copt113 * copt16 * copt664;
    Real copt845 = 1. * copt16 * copt91;
    Real copt850 = copt373 * copt71;
    Real copt851 = copt308 * copt63;
    Real copt852 = copt433 + copt62 + copt64 + copt67 + copt68 + copt727 +
                   copt728 + copt729 + copt850 + copt851;
    Real copt853 = -1. * copt113 * copt16 * copt852;
    Real copt1018 = copt320 * copt368;
    Real copt1019 = copt304 * copt370;
    Real copt1020 = copt1018 + copt1019;
    Real copt1021 = -1. * copt1020 * copt113 * copt16;
    Real copt1097 = -1. * copt5;
    Real copt400 = copt399 * copt49;
    Real copt405 = copt404 * copt57;
    Real copt407 = copt400 + copt401 + copt402 + copt403 + copt405 + copt406 +
                   copt44 + copt45 + copt47 + copt50;
    Real copt408 = -1. * copt113 * copt16 * copt407;
    Real copt1292 = copt1 * copt120;
    Real copt1293 = copt1292 + copt284 + copt352 + copt387;
    Real copt1299 = 1. * copt1293 * copt16;
    Real copt1300 = 0.5 * copt113 * copt410;
    Real copt1301 = copt1299 + copt1300;
    Real copt1305 = -1. * copt113 * copt16 * copt417;
    Real copt1306 = copt1301 * copt79;
    Real copt1307 = copt1305 + copt1306;
    Real copt682 = copt413 * copt66;
    Real copt683 = copt306 * copt75;
    Real copt684 = copt512 + copt679 + copt680 + copt681 + copt682 + copt683 +
                   copt72 + copt73 + copt74 + copt76;
    Real copt685 = -1. * copt113 * copt16 * copt684;
    Real copt687 = 1. * copt104 * copt16;
    Real copt862 = copt399 * copt43;
    Real copt863 = copt404 * copt53;
    Real copt864 = copt413 * copt63;
    Real copt865 = copt307 + copt862 + copt863 + copt864;
    Real copt866 = -1. * copt113 * copt16 * copt865;
    Real copt1034 = copt310 * copt399;
    Real copt1035 = copt323 * copt404;
    Real copt1036 = copt1034 + copt1035;
    Real copt1037 = -1. * copt1036 * copt113 * copt16;
    Real copt1200 = copt373 * copt413;
    Real copt1201 = copt306 * copt308;
    Real copt1202 = copt1200 + copt1201;
    Real copt1203 = -1. * copt113 * copt1202 * copt16;
    Real copt1276 = -1. * copt1;
    Real copt1071 = -1. * copt3;
    Real copt437 = copt202 * copt38;
    Real copt439 = copt438 * copt57;
    Real copt440 = copt437 + copt439;
    Real copt1453 = 1. * copt434 * copt79;
    Real copt1454 = -1. * copt113 * copt444;
    Real copt1455 = copt1453 + copt1454;
    Real copt703 = copt202 * copt32;
    Real copt704 = copt33 + copt34 + copt36 + copt39 + copt703;
    Real copt883 = copt438 * copt53;
    Real copt884 = copt401 + copt402 + copt403 + copt406 + copt883;
    Real copt1047 = copt202 * copt304;
    Real copt1048 = copt323 * copt438;
    Real copt1049 = copt1047 + copt1048;
    Real copt1212 = copt1097 + copt55;
    Real copt1217 = copt202 * copt368;
    Real copt1218 = copt1217 + copt361 + copt465 + copt466 + copt467;
    Real copt1376 = copt3 + copt93;
    Real copt1381 = copt399 * copt438;
    Real copt1382 = copt1381 + copt52 + copt54 + copt58 + copt59;
    Real copt231 = copt230 + copt5;
    Real copt460 = copt230 + copt55;
    Real copt1549 = copt1 * copt460;
    Real copt1550 = copt1549 + copt401 + copt455 + copt457;
    Real copt469 = copt38 * copt468;
    Real copt470 = copt361 + copt465 + copt466 + copt467 + copt469;
    Real copt1551 = 1. * copt1550 * copt79;
    Real copt1552 = -1. * copt113 * copt474;
    Real copt1553 = copt1551 + copt1552;
    Real copt713 = copt32 * copt468;
    Real copt714 = copt239 * copt75;
    Real copt715 = copt713 + copt714;
    Real copt896 = copt239 * copt71;
    Real copt897 = copt72 + copt73 + copt74 + copt76 + copt896;
    Real copt1057 = copt122 + copt5;
    Real copt1062 = copt304 * copt468;
    Real copt1063 = copt1062 + copt33 + copt34 + copt36 + copt39;
    Real copt1227 = copt368 * copt468;
    Real copt1228 = copt239 * copt308;
    Real copt1229 = copt1227 + copt1228;
    Real copt1390 = copt1276 + copt30;
    Real copt1395 = copt239 * copt413;
    Real copt1396 = copt1395 + copt433 + copt727 + copt728 + copt729;
    Real copt1526 = copt239 * copt438;
    Real copt490 = copt10 + copt93;
    Real copt1651 = copt1 * copt490;
    Real copt1652 = copt1651 + copt361 + copt484 + copt487;
    Real copt495 = copt168 * copt57;
    Real copt496 = copt495 + copt52 + copt54 + copt58 + copt59;
    Real copt1653 = 1. * copt1652 * copt79;
    Real copt1654 = -1. * copt113 * copt501;
    Real copt1655 = copt1653 + copt1654;
    Real copt730 = copt499 * copt75;
    Real copt731 = copt433 + copt727 + copt728 + copt729 + copt730;
    Real copt906 = copt168 * copt53;
    Real copt907 = copt499 * copt71;
    Real copt908 = copt906 + copt907;
    Real copt1072 = copt1071 + copt35;
    Real copt1077 = copt168 * copt323;
    Real copt1078 = copt1077 + copt401 + copt402 + copt403 + copt406;
    Real copt1237 = copt1 + copt103;
    Real copt1242 = copt308 * copt499;
    Real copt1243 = copt1242 + copt72 + copt73 + copt74 + copt76;
    Real copt1405 = copt168 * copt399;
    Real copt1406 = copt413 * copt499;
    Real copt1407 = copt1405 + copt1406;
    Real copt205 = copt204 + copt3;
    Real copt1636 = copt168 * copt468;
    Real copt1525 = copt202 * copt499;
    Real copt519 = copt26 * copt499;
    Real copt520 = copt239 * copt49;
    Real copt521 = copt519 + copt520;
    Real copt744 = copt22 * copt499;
    Real copt747 = copt562 + copt743 + copt744 + copt745 + copt746;
    Real copt920 = copt239 * copt43;
    Real copt921 = copt44 + copt45 + copt47 + copt50 + copt920;
    Real copt1087 = copt320 * copt499;
    Real copt1088 = copt239 * copt310;
    Real copt1089 = copt1087 + copt1088;
    Real copt1251 = copt108 + copt5;
    Real copt1256 = copt370 * copt499;
    Real copt1257 = copt1256 + copt20 + copt23 + copt27 + copt28;
    Real copt1415 = copt1071 + copt24;
    Real copt1420 = copt239 * copt404;
    Real copt1421 = copt1420 + copt537 + copt572 + copt573 + copt574;
    Real copt1527 = copt1525 + copt1526;
    Real copt1627 = copt1097 + copt13;
    Real copt540 = copt108 + copt13;
    Real copt545 = copt168 * copt26;
    Real copt546 = copt20 + copt23 + copt27 + copt28 + copt545;
    Real copt756 = copt168 * copt22;
    Real copt757 = copt438 * copt66;
    Real copt758 = copt756 + copt757;
    Real copt933 = copt438 * copt63;
    Real copt934 = copt512 + copt679 + copt680 + copt681 + copt933;
    Real copt1098 = copt1097 + copt46;
    Real copt1103 = copt168 * copt320;
    Real copt1104 = copt1103 + copt562 + copt743 + copt745 + copt746;
    Real copt1266 = copt168 * copt370;
    Real copt1267 = copt373 * copt438;
    Real copt1268 = copt1266 + copt1267;
    Real copt1429 = copt1 + copt90;
    Real copt1434 = copt306 * copt438;
    Real copt1435 = copt1434 + copt62 + copt64 + copt67 + copt68;
    Real copt1637 = copt1526 + copt1636;
    Real copt1737 = copt1276 + copt7;
    Real copt567 = copt204 + copt24;
    Real copt575 = copt468 * copt49;
    Real copt576 = copt537 + copt572 + copt573 + copt574 + copt575;
    Real copt770 = copt202 * copt66;
    Real copt771 = copt62 + copt64 + copt67 + copt68 + copt770;
    Real copt943 = copt43 * copt468;
    Real copt944 = copt202 * copt63;
    Real copt945 = copt943 + copt944;
    Real copt1112 = copt119 + copt3;
    Real copt1117 = copt310 * copt468;
    Real copt1118 = copt1117 + copt44 + copt45 + copt47 + copt50;
    Real copt1277 = copt1276 + copt19;
    Real copt1282 = copt202 * copt373;
    Real copt1283 = copt1282 + copt512 + copt679 + copt680 + copt681;
    Real copt1444 = copt404 * copt468;
    Real copt1445 = copt202 * copt306;
    Real copt1446 = copt1444 + copt1445;
    Real copt1543 = copt10 + copt1071;
    Real copt1645 = copt1 + copt485;
    Real copt1746 = copt1525 + copt1636;
    tMatrix12 out;
    out(0, 0) =
        -(copt18 * copt183 * copt185 * copt188) -
        (copt117 * copt188 * copt192 * copt194) / 2. +
        copt117 * copt18 *
            (-1. * copt126 * copt131 * copt16 -
             2. * copt113 * copt131 * copt168 + copt131 * copt176 -
             1. * copt113 * copt16 *
                 (2 * copt26 * copt38 + 2 * copt49 * copt57) +
             copt79 *
                 (-4. * copt1 * copt13 * copt24 +
                  1. * copt13 * copt24 * copt30 + 4. * copt1 * copt13 * copt35 +
                  4. * copt1 * copt10 * copt46 - 4. * copt1 * copt35 * copt46 -
                  4. * copt1 * copt10 * copt55 + 1. * copt10 * copt19 * copt55 +
                  4. * copt1 * copt24 * copt55 + 3. * copt13 * copt24 * copt7 -
                  3. * copt13 * copt35 * copt7 - 3. * copt10 * copt46 * copt7 +
                  4. * copt35 * copt46 * copt7 + 3. * copt10 * copt55 * copt7 -
                  4. * copt24 * copt55 * copt7 +
                  copt3 * (copt107 + 1. * copt13 * copt19 -
                           1. * copt13 * copt30 + 1. * copt30 * copt46 -
                           1. * copt46 * copt7 + 1. * copt55 * copt7) +
                  copt84 + copt86 +
                  copt5 * (1. * copt19 * copt35 + 1. * copt24 * copt7 -
                           1. * copt35 * copt7 + copt89 +
                           copt10 * (1. * copt30 + copt90))));
    out(0, 1) =
        -(copt18 * copt185 * copt188 * copt221) -
        (copt117 * copt188 * copt194 * copt225) / 2. +
        copt117 * copt18 *
            (-1. * copt111 * copt131 * copt16 -
             2. * copt113 * copt131 * copt202 + copt176 * copt214 -
             1. * copt113 * copt16 * (copt198 + copt43 * copt71) +
             (1. * copt111 * copt168 + 2. * copt126 * copt205) * copt79);
    out(0, 2) = -(copt18 * copt185 * copt188 * copt252) -
                (copt117 * copt188 * copt194 * copt256) / 2. +
                copt117 * copt18 *
                    (-2. * copt113 * copt131 * copt239 + copt176 * copt245 -
                     1. * copt113 * copt16 * (copt236 + copt32 * copt66) -
                     1. * copt131 * copt16 * copt97 +
                     copt79 * (2. * copt126 * copt231 + 1. * copt168 * copt97));
    out(0, 3) =
        -0.5 * (copt117 * copt188 * copt194 * copt316) -
        copt18 * copt185 * copt188 * copt333 +
        copt117 * copt18 *
            (-1. * copt131 * copt16 * copt263 -
             1. * copt113 * copt16 * (copt305 + copt307 + copt309 + copt311) -
             1. * copt113 * copt131 * copt316 + copt176 * copt326 +
             (3. * copt1 * copt13 * copt24 - 1. * copt13 * copt24 * copt30 -
              3. * copt1 * copt13 * copt35 + 1. * copt13 * copt19 * copt35 -
              3. * copt1 * copt10 * copt46 + 1. * copt10 * copt30 * copt46 +
              4. * copt1 * copt35 * copt46 + 3. * copt1 * copt10 * copt55 -
              1. * copt10 * copt19 * copt55 - 4. * copt1 * copt24 * copt55 -
              2. * copt13 * copt24 * copt7 + 2. * copt13 * copt35 * copt7 +
              2. * copt10 * copt46 * copt7 - 4. * copt35 * copt46 * copt7 -
              2. * copt10 * copt55 * copt7 + 4. * copt24 * copt55 * copt7 +
              copt5 * (copt10 * (copt103 + 1. * copt19) + 1. * copt1 * copt24 +
                       copt282 + copt284 + 1. * copt24 * copt30 -
                       2. * copt24 * copt7 + 2. * copt35 * copt7) +
              copt3 * (copt292 + copt293 + copt296 + 1. * copt13 * copt30 +
                       1. * copt1 * copt55 + 1. * copt19 * copt55 +
                       2. * copt46 * copt7 - 2. * copt55 * copt7)) *
                 copt79);
    out(0, 4) = -0.5 * (copt117 * copt188 * copt194 * copt365) -
                copt18 * copt185 * copt188 * copt382 +
                copt117 * copt18 *
                    (-1. * copt131 * copt16 * copt340 + copt363 -
                     1. * copt113 * copt131 * copt365 + copt176 * copt375 +
                     (1. * copt123 * copt173 + 1. * copt168 * copt340 +
                      1. * copt126 * (-2. * copt3 + copt345)) *
                         copt79);
    out(0, 5) = -0.5 * (copt117 * copt188 * copt194 * copt410) -
                copt18 * copt185 * copt188 * copt424 +
                copt117 * copt18 *
                    (-1. * copt131 * copt16 * copt389 + copt408 -
                     1. * copt113 * copt131 * copt410 + copt176 * copt417 +
                     (1. * copt120 * copt173 + 1. * copt168 * copt389 +
                      1. * copt126 * (copt394 - 2. * copt5)) *
                         copt79);
    out(0, 6) = -(copt18 * copt185 * copt188 * copt450) +
                copt117 * copt18 *
                    (-1. * copt131 * copt16 * copt434 -
                     1. * copt113 * copt16 * copt440 + copt176 * copt444 +
                     1. * copt168 * copt434 * copt79);
    out(0, 7) =
        -(copt18 * copt185 * copt188 * copt480) +
        copt117 * copt18 *
            (-1. * copt131 * copt16 * copt458 -
             1. * copt113 * copt16 * copt470 + copt176 * copt474 +
             (1. * copt168 * copt458 + 1. * copt173 * copt460) * copt79);
    out(0, 8) =
        -(copt18 * copt185 * copt188 * copt507) +
        copt117 * copt18 *
            (-1. * copt131 * copt16 * copt488 -
             1. * copt113 * copt16 * copt496 + copt176 * copt501 +
             (1. * copt168 * copt488 + 1. * copt173 * copt490) * copt79);
    out(0, 9) = -(copt18 * copt185 * copt188 * copt531) +
                copt117 * copt18 *
                    (-1. * copt131 * copt16 * copt516 -
                     1. * copt113 * copt16 * copt521 + copt176 * copt525 +
                     1. * copt168 * copt516 * copt79);
    out(0, 10) =
        -(copt18 * copt185 * copt188 * copt556) +
        copt117 * copt18 *
            (-1. * copt131 * copt16 * copt538 -
             1. * copt113 * copt16 * copt546 + copt176 * copt550 +
             (1. * copt168 * copt538 + 1. * copt173 * copt540) * copt79);
    out(0, 11) =
        -(copt18 * copt185 * copt188 * copt586) +
        copt117 * copt18 *
            (-1. * copt131 * copt16 * copt565 -
             1. * copt113 * copt16 * copt576 + copt176 * copt580 +
             (1. * copt168 * copt565 + 1. * copt173 * copt567) * copt79);
    out(1, 0) =
        -(copt18 * copt183 * copt185 * copt606) -
        (copt117 * copt192 * copt194 * copt606) / 2. +
        copt117 * copt18 *
            (-1. * copt126 * copt16 * copt214 -
             1. * copt113 * copt192 * copt214 -
             1. * copt113 * copt16 * (copt198 + copt22 * copt38) +
             copt131 * copt600 +
             (1. * copt111 * copt192 + 1. * copt126 * copt202) * copt79);
    out(1, 1) =
        -(copt18 * copt185 * copt221 * copt606) -
        (copt117 * copt194 * copt225 * copt606) / 2. +
        copt117 * copt18 *
            (-1. * copt111 * copt16 * copt214 -
             1. * copt113 * copt214 * copt225 + copt214 * copt600 -
             1. * copt113 * copt16 *
                 (2 * copt22 * copt32 + 2 * copt66 * copt75) +
             (1. * copt111 * copt202 + 1. * copt111 * copt225 + copt618) *
                 copt79);
    out(1, 2) = -(copt18 * copt185 * copt252 * copt606) -
                (copt117 * copt194 * copt256 * copt606) / 2. +
                copt117 * copt18 *
                    (-1. * copt113 * copt214 * copt256 + copt245 * copt600 +
                     copt631 - 1. * copt16 * copt214 * copt97 +
                     copt79 * (1. * copt111 * copt256 + 1. * copt202 * copt97));
    out(1, 3) =
        -0.5 * (copt117 * copt194 * copt316 * copt606) -
        copt18 * copt185 * copt333 * copt606 +
        copt117 * copt18 *
            (-1. * copt16 * copt214 * copt263 -
             1. * copt113 * copt214 * copt316 + copt326 * copt600 + copt647 +
             (1. * copt202 * copt263 + 1. * copt111 * copt316 + copt649) *
                 copt79);
    out(1, 4) =
        -0.5 * (copt117 * copt194 * copt365 * copt606) -
        copt18 * copt185 * copt382 * copt606 +
        copt117 * copt18 *
            (-1. * copt16 * copt214 * copt340 -
             1. * copt113 * copt214 * copt365 + copt375 * copt600 + copt665 +
             (1. * copt202 * copt340 + 1. * copt111 * copt365 + copt669) *
                 copt79);
    out(1, 5) =
        -0.5 * (copt117 * copt194 * copt410 * copt606) -
        copt18 * copt185 * copt424 * copt606 +
        copt117 * copt18 *
            (-1. * copt16 * copt214 * copt389 -
             1. * copt113 * copt214 * copt410 + copt417 * copt600 + copt685 +
             (1. * copt202 * copt389 + 1. * copt111 * copt410 + copt687) *
                 copt79);
    out(1, 6) = -(copt18 * copt185 * copt450 * copt606) +
                copt117 * copt18 *
                    (-1. * copt16 * copt214 * copt434 + copt444 * copt600 -
                     1. * copt113 * copt16 * copt704 +
                     (1. * copt16 * copt431 + 1. * copt202 * copt434) * copt79);
    out(1, 7) =
        -(copt18 * copt185 * copt480 * copt606) +
        copt117 * copt18 *
            (-1. * copt16 * copt214 * copt458 + copt474 * copt600 -
             1. * copt113 * copt16 * copt715 + 1. * copt202 * copt458 * copt79);
    out(1, 8) = -(copt18 * copt185 * copt507 * copt606) +
                copt117 * copt18 *
                    (-1. * copt16 * copt214 * copt488 + copt501 * copt600 -
                     1. * copt113 * copt16 * copt731 +
                     (1. * copt16 * copt486 + 1. * copt202 * copt488) * copt79);
    out(1, 9) = -(copt18 * copt185 * copt531 * copt606) +
                copt117 * copt18 *
                    (-1. * copt16 * copt214 * copt516 + copt525 * copt600 -
                     1. * copt113 * copt16 * copt747 +
                     (1. * copt16 * copt514 + 1. * copt202 * copt516) * copt79);
    out(1, 10) =
        -(copt18 * copt185 * copt556 * copt606) +
        copt117 * copt18 *
            (-1. * copt16 * copt214 * copt538 + copt550 * copt600 -
             1. * copt113 * copt16 * copt758 + 1. * copt202 * copt538 * copt79);
    out(1, 11) =
        -(copt18 * copt185 * copt586 * copt606) +
        copt117 * copt18 *
            (-1. * copt16 * copt214 * copt565 + copt580 * copt600 -
             1. * copt113 * copt16 * copt771 +
             (1. * copt16 * copt560 + 1. * copt202 * copt565) * copt79);
    out(2, 0) = -(copt18 * copt183 * copt185 * copt795) -
                (copt117 * copt192 * copt194 * copt795) / 2. +
                copt117 * copt18 *
                    (-1. * copt126 * copt16 * copt245 -
                     1. * copt113 * copt192 * copt245 -
                     1. * copt113 * copt16 * (copt236 + copt49 * copt53) +
                     copt131 * copt789 +
                     copt79 * (1. * copt126 * copt239 + 1. * copt192 * copt97));
    out(2, 1) =
        -(copt18 * copt185 * copt221 * copt795) -
        (copt117 * copt194 * copt225 * copt795) / 2. +
        copt117 * copt18 *
            (-1. * copt111 * copt16 * copt245 -
             1. * copt113 * copt225 * copt245 + copt631 + copt214 * copt789 +
             copt79 * (1. * copt111 * copt239 + 1. * copt225 * copt97));
    out(2, 2) = -(copt18 * copt185 * copt252 * copt795) -
                (copt117 * copt194 * copt256 * copt795) / 2. +
                copt117 * copt18 *
                    (-1. * copt113 * copt245 * copt256 -
                     1. * copt113 * copt16 *
                         (2 * copt43 * copt53 + 2 * copt63 * copt71) +
                     copt245 * copt789 - 1. * copt16 * copt245 * copt97 +
                     copt79 * (copt618 + 1. * copt239 * copt97 +
                               1. * copt256 * copt97));
    out(2, 3) =
        -0.5 * (copt117 * copt194 * copt316 * copt795) -
        copt18 * copt185 * copt333 * copt795 +
        copt117 * copt18 *
            (-1. * copt16 * copt245 * copt263 -
             1. * copt113 * copt245 * copt316 + copt326 * copt789 + copt836 +
             copt79 *
                 (1. * copt239 * copt263 + copt828 + 1. * copt316 * copt97));
    out(2, 4) =
        -0.5 * (copt117 * copt194 * copt365 * copt795) -
        copt18 * copt185 * copt382 * copt795 +
        copt117 * copt18 *
            (-1. * copt16 * copt245 * copt340 -
             1. * copt113 * copt245 * copt365 + copt375 * copt789 + copt853 +
             copt79 *
                 (1. * copt239 * copt340 + copt845 + 1. * copt365 * copt97));
    out(2, 5) =
        -0.5 * (copt117 * copt194 * copt410 * copt795) -
        copt18 * copt185 * copt424 * copt795 +
        copt117 * copt18 *
            (-1. * copt16 * copt245 * copt389 -
             1. * copt113 * copt245 * copt410 + copt417 * copt789 + copt866 +
             copt79 *
                 (1. * copt239 * copt389 + copt669 + 1. * copt410 * copt97));
    out(2, 6) = -(copt18 * copt185 * copt450 * copt795) +
                copt117 * copt18 *
                    (-1. * copt16 * copt245 * copt434 + copt444 * copt789 +
                     (1. * copt16 * copt429 + 1. * copt239 * copt434) * copt79 -
                     1. * copt113 * copt16 * copt884);
    out(2, 7) = -(copt18 * copt185 * copt480 * copt795) +
                copt117 * copt18 *
                    (-1. * copt16 * copt245 * copt458 + copt474 * copt789 +
                     (1. * copt16 * copt454 + 1. * copt239 * copt458) * copt79 -
                     1. * copt113 * copt16 * copt897);
    out(2, 8) =
        -(copt18 * copt185 * copt507 * copt795) +
        copt117 * copt18 *
            (-1. * copt16 * copt245 * copt488 + copt501 * copt789 +
             1. * copt239 * copt488 * copt79 - 1. * copt113 * copt16 * copt908);
    out(2, 9) = -(copt18 * copt185 * copt531 * copt795) +
                copt117 * copt18 *
                    (-1. * copt16 * copt245 * copt516 + copt525 * copt789 +
                     (1. * copt16 * copt510 + 1. * copt239 * copt516) * copt79 -
                     1. * copt113 * copt16 * copt921);
    out(2, 10) =
        -(copt18 * copt185 * copt556 * copt795) +
        copt117 * copt18 *
            (-1. * copt16 * copt245 * copt538 + copt550 * copt789 +
             (1. * copt16 * copt535 + 1. * copt239 * copt538) * copt79 -
             1. * copt113 * copt16 * copt934);
    out(2, 11) =
        -(copt18 * copt185 * copt586 * copt795) +
        copt117 * copt18 *
            (-1. * copt16 * copt245 * copt565 + copt580 * copt789 +
             1. * copt239 * copt565 * copt79 - 1. * copt113 * copt16 * copt945);
    out(3, 0) =
        copt117 * copt18 *
            (-1. * copt126 * copt16 * copt326 -
             1. * copt113 * copt192 * copt326 -
             1. * copt113 * copt16 *
                 (copt305 + copt311 + copt320 * copt38 + copt323 * copt49) +
             (1. * copt192 * copt263 + 0.5 * copt126 * copt316 + copt669) *
                 copt79 +
             copt131 * copt964) -
        copt18 * copt183 * copt185 * copt970 -
        (copt117 * copt192 * copt194 * copt970) / 2.;
    out(3, 1) =
        copt117 * copt18 *
            (-1. * copt111 * copt16 * copt326 -
             1. * copt113 * copt225 * copt326 + copt647 +
             (1. * copt225 * copt263 + 0.5 * copt111 * copt316 + copt649) *
                 copt79 +
             copt214 * copt964) -
        copt18 * copt185 * copt221 * copt970 -
        (copt117 * copt194 * copt225 * copt970) / 2.;
    out(3, 2) = copt117 * copt18 *
                    (-1. * copt113 * copt256 * copt326 + copt836 +
                     copt245 * copt964 - 1. * copt16 * copt326 * copt97 +
                     copt79 * (1. * copt256 * copt263 + copt828 +
                               0.5 * copt316 * copt97)) -
                copt18 * copt185 * copt252 * copt970 -
                (copt117 * copt194 * copt256 * copt970) / 2.;
    out(3, 3) =
        copt117 * copt18 *
            (-1. * copt113 * copt16 *
                 (2 * copt304 * copt320 + 2 * copt310 * copt323) -
             1. * copt16 * copt263 * copt326 -
             1. * copt113 * copt316 * copt326 +
             (1.5 * copt263 * copt316 + copt618) * copt79 + copt326 * copt964) -
        (copt117 * copt194 * copt316 * copt970) / 2. -
        copt18 * copt185 * copt333 * copt970;
    out(3, 4) =
        copt117 * copt18 *
            (copt1021 - 1. * copt16 * copt326 * copt340 -
             1. * copt113 * copt326 * copt365 +
             (0.5 * copt316 * copt340 + 1. * copt263 * copt365) * copt79 +
             copt375 * copt964) -
        (copt117 * copt194 * copt365 * copt970) / 2. -
        copt18 * copt185 * copt382 * copt970;
    out(3, 5) =
        copt117 * copt18 *
            (copt1037 - 1. * copt16 * copt326 * copt389 -
             1. * copt113 * copt326 * copt410 +
             (0.5 * copt316 * copt389 + 1. * copt263 * copt410) * copt79 +
             copt417 * copt964) -
        (copt117 * copt194 * copt410 * copt970) / 2. -
        copt18 * copt185 * copt424 * copt970;
    out(3, 6) = copt117 * copt18 *
                    (-1. * copt1049 * copt113 * copt16 -
                     1. * copt16 * copt326 * copt434 +
                     0.5 * copt316 * copt434 * copt79 + copt444 * copt964) -
                copt18 * copt185 * copt450 * copt970;
    out(3, 7) =
        copt117 * copt18 *
            (-1. * copt1063 * copt113 * copt16 -
             1. * copt16 * copt326 * copt458 +
             (1. * copt1057 * copt16 + 0.5 * copt316 * copt458) * copt79 +
             copt474 * copt964) -
        copt18 * copt185 * copt480 * copt970;
    out(3, 8) =
        copt117 * copt18 *
            (-1. * copt1078 * copt113 * copt16 -
             1. * copt16 * copt326 * copt488 +
             (1. * copt1072 * copt16 + 0.5 * copt316 * copt488) * copt79 +
             copt501 * copt964) -
        copt18 * copt185 * copt507 * copt970;
    out(3, 9) = copt117 * copt18 *
                    (-1. * copt1089 * copt113 * copt16 -
                     1. * copt16 * copt326 * copt516 +
                     0.5 * copt316 * copt516 * copt79 + copt525 * copt964) -
                copt18 * copt185 * copt531 * copt970;
    out(3, 10) =
        copt117 * copt18 *
            (-1. * copt1104 * copt113 * copt16 -
             1. * copt16 * copt326 * copt538 +
             (1. * copt1098 * copt16 + 0.5 * copt316 * copt538) * copt79 +
             copt550 * copt964) -
        copt18 * copt185 * copt556 * copt970;
    out(3, 11) =
        copt117 * copt18 *
            (-1. * copt1118 * copt113 * copt16 -
             1. * copt16 * copt326 * copt565 +
             (1. * copt1112 * copt16 + 0.5 * copt316 * copt565) * copt79 +
             copt580 * copt964) -
        copt18 * copt185 * copt586 * copt970;
    out(4, 0) =
        -(copt1140 * copt18 * copt183 * copt185) -
        (copt1140 * copt117 * copt192 * copt194) / 2. +
        copt117 * copt18 *
            (copt1134 * copt131 + copt363 - 1. * copt126 * copt16 * copt375 -
             1. * copt113 * copt192 * copt375 +
             (1. * copt123 * copt16 + 1. * copt192 * copt340 +
              0.5 * copt126 * copt365) *
                 copt79);
    out(4, 1) =
        -(copt1140 * copt18 * copt185 * copt221) -
        (copt1140 * copt117 * copt194 * copt225) / 2. +
        copt117 * copt18 *
            (copt1134 * copt214 - 1. * copt111 * copt16 * copt375 -
             1. * copt113 * copt225 * copt375 + copt665 +
             (1. * copt225 * copt340 + 0.5 * copt111 * copt365 + copt669) *
                 copt79);
    out(4, 2) = -(copt1140 * copt18 * copt185 * copt252) -
                (copt1140 * copt117 * copt194 * copt256) / 2. +
                copt117 * copt18 *
                    (copt1134 * copt245 - 1. * copt113 * copt256 * copt375 +
                     copt853 - 1. * copt16 * copt375 * copt97 +
                     copt79 * (1. * copt256 * copt340 + copt845 +
                               0.5 * copt365 * copt97));
    out(4, 3) =
        -0.5 * (copt1140 * copt117 * copt194 * copt316) -
        copt1140 * copt18 * copt185 * copt333 +
        copt117 * copt18 *
            (copt1021 + copt1134 * copt326 - 1. * copt16 * copt263 * copt375 -
             1. * copt113 * copt316 * copt375 +
             (1. * copt316 * copt340 + 0.5 * copt263 * copt365) * copt79);
    out(4, 4) = -0.5 * (copt1140 * copt117 * copt194 * copt365) -
                copt1140 * copt18 * copt185 * copt382 +
                copt117 * copt18 *
                    (-1. * copt113 * copt16 *
                         (2 * copt368 * copt370 + 2 * copt308 * copt373) +
                     copt1134 * copt375 - 1. * copt16 * copt340 * copt375 -
                     1. * copt113 * copt365 * copt375 +
                     (1.5 * copt340 * copt365 + copt618) * copt79);
    out(4, 5) =
        -0.5 * (copt1140 * copt117 * copt194 * copt410) -
        copt1140 * copt18 * copt185 * copt424 +
        copt117 * copt18 *
            (copt1203 - 1. * copt16 * copt375 * copt389 -
             1. * copt113 * copt375 * copt410 + copt1134 * copt417 +
             (0.5 * copt365 * copt389 + 1. * copt340 * copt410) * copt79);
    out(4, 6) =
        -(copt1140 * copt18 * copt185 * copt450) +
        copt117 * copt18 *
            (-1. * copt113 * copt1218 * copt16 -
             1. * copt16 * copt375 * copt434 + copt1134 * copt444 +
             (1. * copt1212 * copt16 + 0.5 * copt365 * copt434) * copt79);
    out(4, 7) = -(copt1140 * copt18 * copt185 * copt480) +
                copt117 * copt18 *
                    (-1. * copt113 * copt1229 * copt16 -
                     1. * copt16 * copt375 * copt458 + copt1134 * copt474 +
                     0.5 * copt365 * copt458 * copt79);
    out(4, 8) =
        -(copt1140 * copt18 * copt185 * copt507) +
        copt117 * copt18 *
            (-1. * copt113 * copt1243 * copt16 -
             1. * copt16 * copt375 * copt488 + copt1134 * copt501 +
             (1. * copt1237 * copt16 + 0.5 * copt365 * copt488) * copt79);
    out(4, 9) =
        -(copt1140 * copt18 * copt185 * copt531) +
        copt117 * copt18 *
            (-1. * copt113 * copt1257 * copt16 -
             1. * copt16 * copt375 * copt516 + copt1134 * copt525 +
             (1. * copt1251 * copt16 + 0.5 * copt365 * copt516) * copt79);
    out(4, 10) = -(copt1140 * copt18 * copt185 * copt556) +
                 copt117 * copt18 *
                     (-1. * copt113 * copt1268 * copt16 -
                      1. * copt16 * copt375 * copt538 + copt1134 * copt550 +
                      0.5 * copt365 * copt538 * copt79);
    out(4, 11) =
        -(copt1140 * copt18 * copt185 * copt586) +
        copt117 * copt18 *
            (-1. * copt113 * copt1283 * copt16 -
             1. * copt16 * copt375 * copt565 + copt1134 * copt580 +
             (1. * copt1277 * copt16 + 0.5 * copt365 * copt565) * copt79);
    out(5, 0) =
        -(copt1307 * copt18 * copt183 * copt185) -
        (copt117 * copt1307 * copt192 * copt194) / 2. +
        copt117 * copt18 *
            (copt1301 * copt131 + copt408 - 1. * copt126 * copt16 * copt417 -
             1. * copt113 * copt192 * copt417 +
             (1. * copt120 * copt16 + 1. * copt1293 * copt192 +
              0.5 * copt126 * copt410) *
                 copt79);
    out(5, 1) =
        -(copt1307 * copt18 * copt185 * copt221) -
        (copt117 * copt1307 * copt194 * copt225) / 2. +
        copt117 * copt18 *
            (copt1301 * copt214 - 1. * copt111 * copt16 * copt417 -
             1. * copt113 * copt225 * copt417 + copt685 +
             (1. * copt1293 * copt225 + 0.5 * copt111 * copt410 + copt687) *
                 copt79);
    out(5, 2) = -(copt1307 * copt18 * copt185 * copt252) -
                (copt117 * copt1307 * copt194 * copt256) / 2. +
                copt117 * copt18 *
                    (copt1301 * copt245 - 1. * copt113 * copt256 * copt417 +
                     copt866 - 1. * copt16 * copt417 * copt97 +
                     copt79 * (1. * copt1293 * copt256 + copt669 +
                               0.5 * copt410 * copt97));
    out(5, 3) =
        -0.5 * (copt117 * copt1307 * copt194 * copt316) -
        copt1307 * copt18 * copt185 * copt333 +
        copt117 * copt18 *
            (copt1037 + copt1301 * copt326 - 1. * copt16 * copt263 * copt417 -
             1. * copt113 * copt316 * copt417 +
             (1. * copt1293 * copt316 + 0.5 * copt263 * copt410) * copt79);
    out(5, 4) =
        -0.5 * (copt117 * copt1307 * copt194 * copt365) -
        copt1307 * copt18 * copt185 * copt382 +
        copt117 * copt18 *
            (copt1203 + copt1301 * copt375 - 1. * copt16 * copt340 * copt417 -
             1. * copt113 * copt365 * copt417 +
             (1. * copt1293 * copt365 + 0.5 * copt340 * copt410) * copt79);
    out(5, 5) =
        -0.5 * (copt117 * copt1307 * copt194 * copt410) -
        copt1307 * copt18 * copt185 * copt424 +
        copt117 * copt18 *
            (-1. * copt113 * copt16 *
                 (2 * copt399 * copt404 + 2 * copt306 * copt413) +
             copt1301 * copt417 - 1. * copt16 * copt389 * copt417 -
             1. * copt113 * copt410 * copt417 +
             (1. * copt1293 * copt410 + 0.5 * copt389 * copt410 + copt618) *
                 copt79);
    out(5, 6) =
        -(copt1307 * copt18 * copt185 * copt450) +
        copt117 * copt18 *
            (-1. * copt113 * copt1382 * copt16 -
             1. * copt16 * copt417 * copt434 + copt1301 * copt444 +
             (1. * copt1376 * copt16 + 0.5 * copt410 * copt434) * copt79);
    out(5, 7) =
        -(copt1307 * copt18 * copt185 * copt480) +
        copt117 * copt18 *
            (-1. * copt113 * copt1396 * copt16 -
             1. * copt16 * copt417 * copt458 + copt1301 * copt474 +
             (1. * copt1390 * copt16 + 0.5 * copt410 * copt458) * copt79);
    out(5, 8) = -(copt1307 * copt18 * copt185 * copt507) +
                copt117 * copt18 *
                    (-1. * copt113 * copt1407 * copt16 -
                     1. * copt16 * copt417 * copt488 + copt1301 * copt501 +
                     0.5 * copt410 * copt488 * copt79);
    out(5, 9) =
        -(copt1307 * copt18 * copt185 * copt531) +
        copt117 * copt18 *
            (-1. * copt113 * copt1421 * copt16 -
             1. * copt16 * copt417 * copt516 + copt1301 * copt525 +
             (1. * copt1415 * copt16 + 0.5 * copt410 * copt516) * copt79);
    out(5, 10) =
        -(copt1307 * copt18 * copt185 * copt556) +
        copt117 * copt18 *
            (-1. * copt113 * copt1435 * copt16 -
             1. * copt16 * copt417 * copt538 + copt1301 * copt550 +
             (1. * copt1429 * copt16 + 0.5 * copt410 * copt538) * copt79);
    out(5, 11) = -(copt1307 * copt18 * copt185 * copt586) +
                 copt117 * copt18 *
                     (-1. * copt113 * copt1446 * copt16 -
                      1. * copt16 * copt417 * copt565 + copt1301 * copt580 +
                      0.5 * copt410 * copt565 * copt79);
    out(6, 0) = -(copt1455 * copt17 * copt183 * copt185) +
                (copt117 * copt1455 * copt18 * copt192) / 2. +
                copt117 * copt17 *
                    (1. * copt131 * copt434 - 1. * copt113 * copt440 -
                     1. * copt126 * copt444);
    out(6, 1) = -(copt1455 * copt17 * copt185 * copt221) +
                (copt117 * copt1455 * copt18 * copt225) / 2. +
                copt117 * copt17 *
                    (1. * copt214 * copt434 - 1. * copt111 * copt444 -
                     1. * copt113 * copt704 + 1. * copt431 * copt79);
    out(6, 2) = -(copt1455 * copt17 * copt185 * copt252) +
                (copt117 * copt1455 * copt18 * copt256) / 2. +
                copt117 * copt17 *
                    (1. * copt245 * copt434 + 1. * copt429 * copt79 -
                     1. * copt113 * copt884 - 1. * copt444 * copt97);
    out(6, 3) = (copt117 * copt1455 * copt18 * copt316) / 2. -
                copt1455 * copt17 * copt185 * copt333 +
                copt117 * copt17 *
                    (-1. * copt1049 * copt113 + 1. * copt326 * copt434 -
                     1. * copt263 * copt444);
    out(6, 4) = (copt117 * copt1455 * copt18 * copt365) / 2. -
                copt1455 * copt17 * copt185 * copt382 +
                copt117 * copt17 *
                    (-1. * copt113 * copt1218 + 1. * copt375 * copt434 -
                     1. * copt340 * copt444 + 1. * copt1212 * copt79);
    out(6, 5) = (copt117 * copt1455 * copt18 * copt410) / 2. -
                copt1455 * copt17 * copt185 * copt424 +
                copt117 * copt17 *
                    (-1. * copt113 * copt1382 + 1. * copt417 * copt434 -
                     1. * copt389 * copt444 + 1. * copt1376 * copt79);
    out(6, 6) = 0. - copt1455 * copt17 * copt185 * copt450;
    out(6, 7) =
        copt117 * copt17 * (-1. * copt444 * copt458 + 1. * copt434 * copt474) -
        copt1455 * copt17 * copt185 * copt480;
    out(6, 8) =
        copt117 * copt17 * (-1. * copt444 * copt488 + 1. * copt434 * copt501) -
        copt1455 * copt17 * copt185 * copt507;
    out(6, 9) = copt117 * copt17 *
                    (-1. * copt113 * copt1527 - 1. * copt444 * copt516 +
                     1. * copt434 * copt525) -
                copt1455 * copt17 * copt185 * copt531;
    out(6, 10) =
        -(copt1455 * copt17 * copt185 * copt556) +
        copt117 * copt17 *
            (-1. * copt113 * copt168 * copt202 - 1. * copt444 * copt538 +
             1. * copt434 * copt550 + 1. * copt231 * copt79);
    out(6, 11) =
        -(copt1455 * copt17 * copt185 * copt586) +
        copt117 * copt17 *
            (-1. * copt113 * copt438 * copt468 - 1. * copt444 * copt565 +
             1. * copt434 * copt580 + 1. * copt1543 * copt79);
    out(7, 0) = -(copt1553 * copt17 * copt183 * copt185) +
                (copt117 * copt1553 * copt18 * copt192) / 2. +
                copt117 * copt17 *
                    (1. * copt131 * copt1550 - 1. * copt113 * copt470 -
                     1. * copt126 * copt474 + 1. * copt460 * copt79);
    out(7, 1) = -(copt1553 * copt17 * copt185 * copt221) +
                (copt117 * copt1553 * copt18 * copt225) / 2. +
                copt117 * copt17 *
                    (1. * copt1550 * copt214 - 1. * copt111 * copt474 -
                     1. * copt113 * copt715);
    out(7, 2) = -(copt1553 * copt17 * copt185 * copt252) +
                (copt117 * copt1553 * copt18 * copt256) / 2. +
                copt117 * copt17 *
                    (1. * copt1550 * copt245 + 1. * copt454 * copt79 -
                     1. * copt113 * copt897 - 1. * copt474 * copt97);
    out(7, 3) = (copt117 * copt1553 * copt18 * copt316) / 2. -
                copt1553 * copt17 * copt185 * copt333 +
                copt117 * copt17 *
                    (-1. * copt1063 * copt113 + 1. * copt1550 * copt326 -
                     1. * copt263 * copt474 + 1. * copt1057 * copt79);
    out(7, 4) = (copt117 * copt1553 * copt18 * copt365) / 2. -
                copt1553 * copt17 * copt185 * copt382 +
                copt117 * copt17 *
                    (-1. * copt113 * copt1229 + 1. * copt1550 * copt375 -
                     1. * copt340 * copt474);
    out(7, 5) = (copt117 * copt1553 * copt18 * copt410) / 2. -
                copt1553 * copt17 * copt185 * copt424 +
                copt117 * copt17 *
                    (-1. * copt113 * copt1396 + 1. * copt1550 * copt417 -
                     1. * copt389 * copt474 + 1. * copt1390 * copt79);
    out(7, 6) =
        -(copt1553 * copt17 * copt185 * copt450) +
        copt117 * copt17 * (1. * copt1550 * copt444 - 1. * copt434 * copt474);
    out(7, 7) =
        copt117 * copt17 * (1. * copt1550 * copt474 - 1. * copt458 * copt474) -
        copt1553 * copt17 * copt185 * copt480;
    out(7, 8) =
        copt117 * copt17 * (-1. * copt474 * copt488 + 1. * copt1550 * copt501) -
        copt1553 * copt17 * copt185 * copt507;
    out(7, 9) =
        -(copt1553 * copt17 * copt185 * copt531) +
        copt117 * copt17 *
            (-1. * copt113 * copt468 * copt499 - 1. * copt474 * copt516 +
             1. * copt1550 * copt525 + 1. * copt1627 * copt79);
    out(7, 10) = copt117 * copt17 *
                     (-1. * copt113 * copt1637 - 1. * copt474 * copt538 +
                      1. * copt1550 * copt550) -
                 copt1553 * copt17 * copt185 * copt556;
    out(7, 11) =
        -(copt1553 * copt17 * copt185 * copt586) +
        copt117 * copt17 *
            (-1. * copt113 * copt202 * copt239 - 1. * copt474 * copt565 +
             1. * copt1550 * copt580 + 1. * copt1645 * copt79);
    out(8, 0) = -(copt1655 * copt17 * copt183 * copt185) +
                (copt117 * copt1655 * copt18 * copt192) / 2. +
                copt117 * copt17 *
                    (1. * copt131 * copt1652 - 1. * copt113 * copt496 -
                     1. * copt126 * copt501 + 1. * copt490 * copt79);
    out(8, 1) = -(copt1655 * copt17 * copt185 * copt221) +
                (copt117 * copt1655 * copt18 * copt225) / 2. +
                copt117 * copt17 *
                    (1. * copt1652 * copt214 - 1. * copt111 * copt501 -
                     1. * copt113 * copt731 + 1. * copt486 * copt79);
    out(8, 2) = -(copt1655 * copt17 * copt185 * copt252) +
                (copt117 * copt1655 * copt18 * copt256) / 2. +
                copt117 * copt17 *
                    (1. * copt1652 * copt245 - 1. * copt113 * copt908 -
                     1. * copt501 * copt97);
    out(8, 3) = (copt117 * copt1655 * copt18 * copt316) / 2. -
                copt1655 * copt17 * copt185 * copt333 +
                copt117 * copt17 *
                    (-1. * copt1078 * copt113 + 1. * copt1652 * copt326 -
                     1. * copt263 * copt501 + 1. * copt1072 * copt79);
    out(8, 4) = (copt117 * copt1655 * copt18 * copt365) / 2. -
                copt1655 * copt17 * copt185 * copt382 +
                copt117 * copt17 *
                    (-1. * copt113 * copt1243 + 1. * copt1652 * copt375 -
                     1. * copt340 * copt501 + 1. * copt1237 * copt79);
    out(8, 5) = (copt117 * copt1655 * copt18 * copt410) / 2. -
                copt1655 * copt17 * copt185 * copt424 +
                copt117 * copt17 *
                    (-1. * copt113 * copt1407 + 1. * copt1652 * copt417 -
                     1. * copt389 * copt501);
    out(8, 6) =
        -(copt1655 * copt17 * copt185 * copt450) +
        copt117 * copt17 * (1. * copt1652 * copt444 - 1. * copt434 * copt501);
    out(8, 7) =
        -(copt1655 * copt17 * copt185 * copt480) +
        copt117 * copt17 * (1. * copt1652 * copt474 - 1. * copt458 * copt501);
    out(8, 8) =
        copt117 * copt17 * (1. * copt1652 * copt501 - 1. * copt488 * copt501) -
        copt1655 * copt17 * copt185 * copt507;
    out(8, 9) =
        -(copt1655 * copt17 * copt185 * copt531) +
        copt117 * copt17 *
            (-1. * copt113 * copt168 * copt239 - 1. * copt501 * copt516 +
             1. * copt1652 * copt525 + 1. * copt205 * copt79);
    out(8, 10) =
        -(copt1655 * copt17 * copt185 * copt556) +
        copt117 * copt17 *
            (-1. * copt113 * copt438 * copt499 - 1. * copt501 * copt538 +
             1. * copt1652 * copt550 + 1. * copt1737 * copt79);
    out(8, 11) = copt117 * copt17 *
                     (-1. * copt113 * copt1746 - 1. * copt501 * copt565 +
                      1. * copt1652 * copt580) -
                 copt1655 * copt17 * copt185 * copt586;
    out(9, 0) = 1. * copt117 * copt131 * copt17 * copt516 -
                1. * copt113 * copt117 * copt17 * copt521 -
                1. * copt117 * copt126 * copt17 * copt525 +
                1. * copt113 * copt17 * copt183 * copt185 * copt525 -
                0.5 * copt113 * copt117 * copt18 * copt192 * copt525 -
                1. * copt17 * copt183 * copt185 * copt516 * copt79 +
                0.5 * copt117 * copt18 * copt192 * copt516 * copt79;
    out(9, 1) = 1. * copt117 * copt17 * copt214 * copt516 -
                1. * copt111 * copt117 * copt17 * copt525 +
                1. * copt113 * copt17 * copt185 * copt221 * copt525 -
                0.5 * copt113 * copt117 * copt18 * copt225 * copt525 -
                1. * copt113 * copt117 * copt17 * copt747 +
                1. * copt117 * copt17 * copt514 * copt79 -
                1. * copt17 * copt185 * copt221 * copt516 * copt79 +
                0.5 * copt117 * copt18 * copt225 * copt516 * copt79;
    out(9, 2) = 1. * copt117 * copt17 * copt245 * copt516 +
                1. * copt113 * copt17 * copt185 * copt252 * copt525 -
                0.5 * copt113 * copt117 * copt18 * copt256 * copt525 +
                1. * copt117 * copt17 * copt510 * copt79 -
                1. * copt17 * copt185 * copt252 * copt516 * copt79 +
                0.5 * copt117 * copt18 * copt256 * copt516 * copt79 -
                1. * copt113 * copt117 * copt17 * copt921 -
                1. * copt117 * copt17 * copt525 * copt97;
    out(9, 3) = -1. * copt1089 * copt113 * copt117 * copt17 +
                1. * copt117 * copt17 * copt326 * copt516 -
                1. * copt117 * copt17 * copt263 * copt525 -
                0.5 * copt113 * copt117 * copt18 * copt316 * copt525 +
                1. * copt113 * copt17 * copt185 * copt333 * copt525 +
                0.5 * copt117 * copt18 * copt316 * copt516 * copt79 -
                1. * copt17 * copt185 * copt333 * copt516 * copt79;
    out(9, 4) = -1. * copt113 * copt117 * copt1257 * copt17 +
                1. * copt117 * copt17 * copt375 * copt516 -
                1. * copt117 * copt17 * copt340 * copt525 -
                0.5 * copt113 * copt117 * copt18 * copt365 * copt525 +
                1. * copt113 * copt17 * copt185 * copt382 * copt525 +
                1. * copt117 * copt1251 * copt17 * copt79 +
                0.5 * copt117 * copt18 * copt365 * copt516 * copt79 -
                1. * copt17 * copt185 * copt382 * copt516 * copt79;
    out(9, 5) = -1. * copt113 * copt117 * copt1421 * copt17 +
                1. * copt117 * copt17 * copt417 * copt516 -
                1. * copt117 * copt17 * copt389 * copt525 -
                0.5 * copt113 * copt117 * copt18 * copt410 * copt525 +
                1. * copt113 * copt17 * copt185 * copt424 * copt525 +
                1. * copt117 * copt1415 * copt17 * copt79 +
                0.5 * copt117 * copt18 * copt410 * copt516 * copt79 -
                1. * copt17 * copt185 * copt424 * copt516 * copt79;
    out(9, 6) = -1. * copt113 * copt117 * copt1527 * copt17 +
                1. * copt117 * copt17 * copt444 * copt516 -
                1. * copt117 * copt17 * copt434 * copt525 +
                1. * copt113 * copt17 * copt185 * copt450 * copt525 -
                1. * copt17 * copt185 * copt450 * copt516 * copt79;
    out(9, 7) = -1. * copt113 * copt117 * copt17 * copt468 * copt499 +
                1. * copt117 * copt17 * copt474 * copt516 -
                1. * copt117 * copt17 * copt458 * copt525 +
                1. * copt113 * copt17 * copt185 * copt480 * copt525 +
                1. * copt117 * copt1627 * copt17 * copt79 -
                1. * copt17 * copt185 * copt480 * copt516 * copt79;
    out(9, 8) = -1. * copt113 * copt117 * copt168 * copt17 * copt239 +
                1. * copt117 * copt17 * copt501 * copt516 -
                1. * copt117 * copt17 * copt488 * copt525 +
                1. * copt113 * copt17 * copt185 * copt507 * copt525 +
                1. * copt117 * copt17 * copt205 * copt79 -
                1. * copt17 * copt185 * copt507 * copt516 * copt79;
    out(9, 9) = 0. + 1. * copt113 * copt17 * copt185 * copt525 * copt531 -
                1. * copt17 * copt185 * copt516 * copt531 * copt79;
    out(9, 10) = -1. * copt117 * copt17 * copt525 * copt538 +
                 1. * copt117 * copt17 * copt516 * copt550 +
                 1. * copt113 * copt17 * copt185 * copt525 * copt556 -
                 1. * copt17 * copt185 * copt516 * copt556 * copt79;
    out(9, 11) = -1. * copt117 * copt17 * copt525 * copt565 +
                 1. * copt117 * copt17 * copt516 * copt580 +
                 1. * copt113 * copt17 * copt185 * copt525 * copt586 -
                 1. * copt17 * copt185 * copt516 * copt586 * copt79;
    out(10, 0) = 1. * copt117 * copt131 * copt17 * copt538 -
                 1. * copt113 * copt117 * copt17 * copt546 -
                 1. * copt117 * copt126 * copt17 * copt550 +
                 1. * copt113 * copt17 * copt183 * copt185 * copt550 -
                 0.5 * copt113 * copt117 * copt18 * copt192 * copt550 -
                 1. * copt17 * copt183 * copt185 * copt538 * copt79 +
                 0.5 * copt117 * copt18 * copt192 * copt538 * copt79 +
                 1. * copt117 * copt17 * copt540 * copt79;
    out(10, 1) = 1. * copt117 * copt17 * copt214 * copt538 -
                 1. * copt111 * copt117 * copt17 * copt550 +
                 1. * copt113 * copt17 * copt185 * copt221 * copt550 -
                 0.5 * copt113 * copt117 * copt18 * copt225 * copt550 -
                 1. * copt113 * copt117 * copt17 * copt758 -
                 1. * copt17 * copt185 * copt221 * copt538 * copt79 +
                 0.5 * copt117 * copt18 * copt225 * copt538 * copt79;
    out(10, 2) = 1. * copt117 * copt17 * copt245 * copt538 +
                 1. * copt113 * copt17 * copt185 * copt252 * copt550 -
                 0.5 * copt113 * copt117 * copt18 * copt256 * copt550 +
                 1. * copt117 * copt17 * copt535 * copt79 -
                 1. * copt17 * copt185 * copt252 * copt538 * copt79 +
                 0.5 * copt117 * copt18 * copt256 * copt538 * copt79 -
                 1. * copt113 * copt117 * copt17 * copt934 -
                 1. * copt117 * copt17 * copt550 * copt97;
    out(10, 3) = -1. * copt1104 * copt113 * copt117 * copt17 +
                 1. * copt117 * copt17 * copt326 * copt538 -
                 1. * copt117 * copt17 * copt263 * copt550 -
                 0.5 * copt113 * copt117 * copt18 * copt316 * copt550 +
                 1. * copt113 * copt17 * copt185 * copt333 * copt550 +
                 1. * copt1098 * copt117 * copt17 * copt79 +
                 0.5 * copt117 * copt18 * copt316 * copt538 * copt79 -
                 1. * copt17 * copt185 * copt333 * copt538 * copt79;
    out(10, 4) = -1. * copt113 * copt117 * copt1268 * copt17 +
                 1. * copt117 * copt17 * copt375 * copt538 -
                 1. * copt117 * copt17 * copt340 * copt550 -
                 0.5 * copt113 * copt117 * copt18 * copt365 * copt550 +
                 1. * copt113 * copt17 * copt185 * copt382 * copt550 +
                 0.5 * copt117 * copt18 * copt365 * copt538 * copt79 -
                 1. * copt17 * copt185 * copt382 * copt538 * copt79;
    out(10, 5) = -1. * copt113 * copt117 * copt1435 * copt17 +
                 1. * copt117 * copt17 * copt417 * copt538 -
                 1. * copt117 * copt17 * copt389 * copt550 -
                 0.5 * copt113 * copt117 * copt18 * copt410 * copt550 +
                 1. * copt113 * copt17 * copt185 * copt424 * copt550 +
                 1. * copt117 * copt1429 * copt17 * copt79 +
                 0.5 * copt117 * copt18 * copt410 * copt538 * copt79 -
                 1. * copt17 * copt185 * copt424 * copt538 * copt79;
    out(10, 6) = -1. * copt113 * copt117 * copt168 * copt17 * copt202 +
                 1. * copt117 * copt17 * copt444 * copt538 -
                 1. * copt117 * copt17 * copt434 * copt550 +
                 1. * copt113 * copt17 * copt185 * copt450 * copt550 +
                 1. * copt117 * copt17 * copt231 * copt79 -
                 1. * copt17 * copt185 * copt450 * copt538 * copt79;
    out(10, 7) = -1. * copt113 * copt117 * copt1637 * copt17 +
                 1. * copt117 * copt17 * copt474 * copt538 -
                 1. * copt117 * copt17 * copt458 * copt550 +
                 1. * copt113 * copt17 * copt185 * copt480 * copt550 -
                 1. * copt17 * copt185 * copt480 * copt538 * copt79;
    out(10, 8) = -1. * copt113 * copt117 * copt17 * copt438 * copt499 +
                 1. * copt117 * copt17 * copt501 * copt538 -
                 1. * copt117 * copt17 * copt488 * copt550 +
                 1. * copt113 * copt17 * copt185 * copt507 * copt550 +
                 1. * copt117 * copt17 * copt1737 * copt79 -
                 1. * copt17 * copt185 * copt507 * copt538 * copt79;
    out(10, 9) = 1. * copt117 * copt17 * copt525 * copt538 -
                 1. * copt117 * copt17 * copt516 * copt550 +
                 1. * copt113 * copt17 * copt185 * copt531 * copt550 -
                 1. * copt17 * copt185 * copt531 * copt538 * copt79;
    out(10, 10) = 0. + 1. * copt113 * copt17 * copt185 * copt550 * copt556 -
                  1. * copt17 * copt185 * copt538 * copt556 * copt79;
    out(10, 11) = -1. * copt117 * copt17 * copt550 * copt565 +
                  1. * copt117 * copt17 * copt538 * copt580 +
                  1. * copt113 * copt17 * copt185 * copt550 * copt586 -
                  1. * copt17 * copt185 * copt538 * copt586 * copt79;
    out(11, 0) = 1. * copt117 * copt131 * copt17 * copt565 -
                 1. * copt113 * copt117 * copt17 * copt576 -
                 1. * copt117 * copt126 * copt17 * copt580 +
                 1. * copt113 * copt17 * copt183 * copt185 * copt580 -
                 0.5 * copt113 * copt117 * copt18 * copt192 * copt580 -
                 1. * copt17 * copt183 * copt185 * copt565 * copt79 +
                 0.5 * copt117 * copt18 * copt192 * copt565 * copt79 +
                 1. * copt117 * copt17 * copt567 * copt79;
    out(11, 1) = 1. * copt117 * copt17 * copt214 * copt565 -
                 1. * copt111 * copt117 * copt17 * copt580 +
                 1. * copt113 * copt17 * copt185 * copt221 * copt580 -
                 0.5 * copt113 * copt117 * copt18 * copt225 * copt580 -
                 1. * copt113 * copt117 * copt17 * copt771 +
                 1. * copt117 * copt17 * copt560 * copt79 -
                 1. * copt17 * copt185 * copt221 * copt565 * copt79 +
                 0.5 * copt117 * copt18 * copt225 * copt565 * copt79;
    out(11, 2) = 1. * copt117 * copt17 * copt245 * copt565 +
                 1. * copt113 * copt17 * copt185 * copt252 * copt580 -
                 0.5 * copt113 * copt117 * copt18 * copt256 * copt580 -
                 1. * copt17 * copt185 * copt252 * copt565 * copt79 +
                 0.5 * copt117 * copt18 * copt256 * copt565 * copt79 -
                 1. * copt113 * copt117 * copt17 * copt945 -
                 1. * copt117 * copt17 * copt580 * copt97;
    out(11, 3) = -1. * copt1118 * copt113 * copt117 * copt17 +
                 1. * copt117 * copt17 * copt326 * copt565 -
                 1. * copt117 * copt17 * copt263 * copt580 -
                 0.5 * copt113 * copt117 * copt18 * copt316 * copt580 +
                 1. * copt113 * copt17 * copt185 * copt333 * copt580 +
                 1. * copt1112 * copt117 * copt17 * copt79 +
                 0.5 * copt117 * copt18 * copt316 * copt565 * copt79 -
                 1. * copt17 * copt185 * copt333 * copt565 * copt79;
    out(11, 4) = -1. * copt113 * copt117 * copt1283 * copt17 +
                 1. * copt117 * copt17 * copt375 * copt565 -
                 1. * copt117 * copt17 * copt340 * copt580 -
                 0.5 * copt113 * copt117 * copt18 * copt365 * copt580 +
                 1. * copt113 * copt17 * copt185 * copt382 * copt580 +
                 1. * copt117 * copt1277 * copt17 * copt79 +
                 0.5 * copt117 * copt18 * copt365 * copt565 * copt79 -
                 1. * copt17 * copt185 * copt382 * copt565 * copt79;
    out(11, 5) = -1. * copt113 * copt117 * copt1446 * copt17 +
                 1. * copt117 * copt17 * copt417 * copt565 -
                 1. * copt117 * copt17 * copt389 * copt580 -
                 0.5 * copt113 * copt117 * copt18 * copt410 * copt580 +
                 1. * copt113 * copt17 * copt185 * copt424 * copt580 +
                 0.5 * copt117 * copt18 * copt410 * copt565 * copt79 -
                 1. * copt17 * copt185 * copt424 * copt565 * copt79;
    out(11, 6) = -1. * copt113 * copt117 * copt17 * copt438 * copt468 +
                 1. * copt117 * copt17 * copt444 * copt565 -
                 1. * copt117 * copt17 * copt434 * copt580 +
                 1. * copt113 * copt17 * copt185 * copt450 * copt580 +
                 1. * copt117 * copt1543 * copt17 * copt79 -
                 1. * copt17 * copt185 * copt450 * copt565 * copt79;
    out(11, 7) = -1. * copt113 * copt117 * copt17 * copt202 * copt239 +
                 1. * copt117 * copt17 * copt474 * copt565 -
                 1. * copt117 * copt17 * copt458 * copt580 +
                 1. * copt113 * copt17 * copt185 * copt480 * copt580 +
                 1. * copt117 * copt1645 * copt17 * copt79 -
                 1. * copt17 * copt185 * copt480 * copt565 * copt79;
    out(11, 8) = -1. * copt113 * copt117 * copt17 * copt1746 +
                 1. * copt117 * copt17 * copt501 * copt565 -
                 1. * copt117 * copt17 * copt488 * copt580 +
                 1. * copt113 * copt17 * copt185 * copt507 * copt580 -
                 1. * copt17 * copt185 * copt507 * copt565 * copt79;
    out(11, 9) = 1. * copt117 * copt17 * copt525 * copt565 -
                 1. * copt117 * copt17 * copt516 * copt580 +
                 1. * copt113 * copt17 * copt185 * copt531 * copt580 -
                 1. * copt17 * copt185 * copt531 * copt565 * copt79;
    out(11, 10) = 1. * copt117 * copt17 * copt550 * copt565 -
                  1. * copt117 * copt17 * copt538 * copt580 +
                  1. * copt113 * copt17 * copt185 * copt556 * copt580 -
                  1. * copt17 * copt185 * copt556 * copt565 * copt79;
    out(11, 11) = 0. + 1. * copt113 * copt17 * copt185 * copt580 * copt586 -
                  1. * copt17 * copt185 * copt565 * copt586 * copt79;
    return out;
}
} // namespace nDihedralAnglesMachine