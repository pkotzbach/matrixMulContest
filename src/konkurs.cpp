#include "CVct.h"

// W ponizszych plikach - impelementacja referencyjna
#include "CMtxRef.h"
//czasy podobne co w tej danej, a dziala na release

#include <stdlib.h>
#include <stdio.h>
#include <iomanip>


// ===================================================================
// FUNKCJE DO POMIARU CZASU
// ===================================================================

#include <sys/timeb.h>
#include <time.h>
#include <math.h>

double mygettime(void) {
    struct _timeb tb;
    _ftime_s(&tb);
    return (double)tb.time + (0.001 * (double)tb.millitm);
}

// ===================================================================
// FUNKCJA OCENY CZASU WYKONANIA
// ===================================================================

// Definiujemy szablon aby latwiej uruchamiac testy dla roznych implementacji
// klasy. Rozne implementacje beda umieszczone w roznych przestrzeniach nazw.
template<typename T>
double test(int SIZE, int ITER_CNT)
{
    // Przykladowe testowe obliczenie macierzowe. Podobne obliczenia beda
    // uzywane do oceny efektywnosci implementacji w konkursie.
    //const int SIZE = 100;
    //const int ITER_CNT = 100;

    T A(SIZE, SIZE, true);
    T B(SIZE, SIZE, true);
    T W(1, 1, false);

    double t1 = mygettime();

    for (int i = 0; i < ITER_CNT; i++)
    {
        B = (A * (0.1 * i) + B * B) * 1.e-4;
        B = -B * ~(A + B);

        //B = A + B;
    }
    W = (B - A);

    double exec_time = mygettime() - t1;

        //W.display();
        //std::cout << "\n";

    return exec_time;
}

void randomCheck()
{
    const int bound = 2000;
    const int SIZE_COM = rand() % bound + 1;
    const int SIZE_Y_1 = rand() % bound + 1;
    const int SIZE_X_2 = rand() % bound + 1;

    MyAlgebra::CMtx My1(SIZE_COM, SIZE_Y_1, false);
    MyRefAlgebra::CMtx Ref1(SIZE_COM, SIZE_Y_1, true);

    MyAlgebra::CMtx My2(SIZE_X_2, SIZE_COM, false);
    MyRefAlgebra::CMtx Ref2(SIZE_X_2, SIZE_COM, true);

    int total_size = SIZE_Y_1 * SIZE_X_2;
    
    int it = 1;
    if (total_size < MyAlgebra::MEDIUM_MATRIX_BOUND) it = 100;
    else if (total_size < MyAlgebra::BIG_MATRIX_BOUND) it = 10;

    std::cout << "SIZE_COM: " << SIZE_COM << ", SIZE_Y_1: " << SIZE_Y_1 << ", SIZE_X_2: " << SIZE_X_2 << ", it: " << it << std::endl;

    for (int i = 0; i < SIZE_COM; i++)
        for (int j = 0; j < SIZE_Y_1; j++) {
            My1[i][j] = Ref1[i][j];
        }

    for (int i = 0; i < SIZE_X_2; i++)
        for (int j = 0; j < SIZE_COM; j++) {
            My2[i][j] = Ref2[i][j];
        }

    //B = (A * (0.1 * i) + B * B) * 1.e-4;
    //B = -B * ~(A + B);

    double t1 = mygettime();
    for (int i = 0; i < it; i++) {
        //Ref2 = (Ref1 * (0.1 * i) + Ref2 * Ref2) * 1.e-4;
        //Ref2 = -Ref2 * ~(Ref1 - Ref2);

        Ref2 = Ref1 * Ref2;

        //Ref2 = Ref2 - Ref1;
        //Ref2 = Ref2 * 1.23123412;
    }
    double exec_time1 = mygettime() - t1;

    t1 = mygettime();
    for (int i = 0; i < it; i++) {
        //My2 = (My1 * (0.1 * i) + My2 * My2) * 1.e-4;
        //My2 = -My2 * ~(My1 - My2);

        My2 = My1 * My2;

        //My2 = My2 - My1;
        //My2 = My2 * 1.23123412;
    }
    double exec_time2 = mygettime() - t1;

    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(10);
    for (int i = 0; i < SIZE_X_2; i++)
        for (int j = 0; j < SIZE_Y_1; j++) {
            if (abs(My2[i][j] - Ref2[i][j]) > MyAlgebra::CMtx::ALG_PRECISION) {
                std::cout << std::endl;
                std::cout << "My2: " << My2[i][j] << ", Ref2: " << Ref2[i][j] << std::endl;
                std::cout << "i: " << i << ", j: " << j << std::endl;
                std::cout << abs(My2[i][j] - Ref2[i][j]);
                throw std::exception();
            }
            //std::cout << abs(My[i][j] - Ref[i][j]) << std::endl;
        }

    std::cout << exec_time1 / exec_time2;
    std::cout << "\n";
}

void checkTime()
{
    const int bound = 1500;
    const int SIZE_X = rand() % bound + 1;
    const int SIZE_Y = SIZE_X;
    MyAlgebra::CMtx My(SIZE_X, SIZE_Y, true);

    const float PREC = 0.0001;

    int it = 1;
    if (SIZE_X * SIZE_X < MyAlgebra::MEDIUM_MATRIX_BOUND) it = 1000;
    else if (SIZE_X * SIZE_X < MyAlgebra::BIG_MATRIX_BOUND) it = 10;

    std::cout << "size: " << SIZE_X << ", it: " << it << std::endl;

    double t1;
    t1 = mygettime();
    for (int i = 0; i < it; i++) {
        //My = (My * (0.83) + My * My) * 1.e-4;
        //My = -My * ~(My - My * 1.3212);

        My = My * My;

        //My = My + My;
        //My = My * 1.23123412;
    }
    double exec_time2 = mygettime() - t1;

    std::cout << exec_time2;
    std::cout << "\n";
}

bool isCorrect()
{
    const int SIZE_X = rand() % 1300 + 1;
    const int SIZE_Y = SIZE_X;
    const float PREC = 0.0000001;
    MyAlgebra::CMtx My(SIZE_X, SIZE_Y, true);
    MyRefAlgebra::CMtx Ref(SIZE_X, SIZE_Y, true);

    std::cout << "isCorrect()\n"
        << "size: " << SIZE_X << std::endl;

    //MyAlgebra::CMtx My2(SIZE_Y, SIZE_X, true);
    //MyRefAlgebra::CMtx Ref2(SIZE_Y, SIZE_X, true);

    for (int i = 0; i < SIZE_X; i++)
        for (int j = 0; j < SIZE_Y; j++) {
            My.set(i, j, Ref[i][j]);
            //My2.set(j, i, Ref2[j][i]);
        }

    //My.display();
    //std::cout << std::endl;
    //Ref.display();
    //std::cout << std::endl;
    //std::cout << std::endl;

    My = (My * (0.83) + My * My) * 1.e-4;
    My = -My * ~(My - My * 1.3212);
    Ref = (Ref * (0.83) + Ref * Ref) * 1.e-4;
    Ref = -Ref * ~(Ref - Ref * 1.3212);

    //My = My * My;
    //Ref = Ref * Ref;

    //My.display();
    //std::cout << std::endl;
    //Ref.display();
    //std::cout << std::endl;

    for (int i = 0; i < SIZE_X; i++)
        for (int j = 0; j < SIZE_Y; j++) {
            if (abs(My[i][j] - Ref[i][j]) > PREC) {
                std::cout << "My: " << My[i][j] << ", Ref: "<< Ref[i][j] << std::endl;
                std::cout << "i: " << i << ", j: " << j << std::endl;
                return false;
            }
            //std::cout << abs(My[i][j] - Ref[i][j]) << std::endl;
        }

    return true;
}

const int TEST = 5; //ilosc testow
const int cases = 5;

int main()
{
    srand(static_cast <unsigned> (time(0)));

    //for (int i = 0; i < 5; i++)
    //    std::cout << isCorrect() << std::endl;

    // for (int i = 0; i < 100; i++)
    //     randomCheck();

    //for (int i = 0; i < 400; i++)
    //    checkTime();

    double t_prog, t_ref;
    const int offset = 127;

    float res[TEST];
    int size_tab[cases] = { 29, 256 + offset, 1024 + offset, 1500, 2048 + offset};
    int iter_tab[cases] = { 10000, 100, 4, 1, 1};

    //all tests
    if (0) {
        float ref_times[cases];
        float test_times[cases];

        for (int j = 0; j < cases; j++) {
            std::cout << "SIZE: " << size_tab[j] << ", ITER: " << iter_tab[j] << std::endl;

            t_ref = 1;
            t_ref = test<MyRefAlgebra::CMtx>(size_tab[j], iter_tab[j]);
            printf("Czas wykonania referencyjny: %7.2lfs\n", t_ref);
            ref_times[j] = t_ref;

            for (int i = 0; i < TEST; ++i) {
                t_prog = test<MyAlgebra::CMtx>(size_tab[j], iter_tab[j]);

                printf("Czas wykonania testowany:    %7.2lfs\n", t_prog);
                printf("Wspolczynnik przyspieszenia Q: %5.2lf\n", t_ref / t_prog);
                res[i] = t_prog;
            }

            float result = 0;
            for (int i = 0; i < TEST; ++i) {
                result += res[i];
            }

            printf("Sredni czas wykonania:       %7.2lfs\n\n", result / TEST);
            test_times[j] = result / TEST;
        }

        //podsumowanie
        for (int j = 0; j < cases; j++) {
            std::cout << "SIZE: " << size_tab[j] << ", ITER: " << iter_tab[j] << std::endl;
            printf("Czas wykonania referencyjny: %7.2lfs\n", ref_times[j]);
            printf("%7.2lfs\n", test_times[j]);
            printf("Srednie Q:                   %7.2lf\n\n", ref_times[j] / test_times[j]);
            }
    }
    
    //one given test
    if (0) {
        int id = 0; //id of test
        std::cout << "SIZE: " << size_tab[id] << ", ITER: " << iter_tab[id] << std::endl;

        t_ref = 1;
        t_ref = test<MyRefAlgebra::CMtx>(size_tab[id], iter_tab[id]);
        printf("Czas wykonania referencyjny: %7.2lfs\n", t_ref);

        for (int i = 0; i < TEST; ++i) {
            t_prog = test<MyAlgebra::CMtx>(size_tab[id], iter_tab[id]);

            printf("Czas wykonania testowany:    %7.2lfs\n", t_prog);
            printf("Wspolczynnik przyspieszenia Q: %5.2lf\n", t_ref / t_prog);
            res[i] = t_prog;
        }

        float result = 0;
        for (int i = 0; i < TEST; ++i) {
            result += res[i];
        }

        printf("Sredni czas wykonania:       %7.2lfs\n\n", result / TEST);
    }

    return 0;
}
