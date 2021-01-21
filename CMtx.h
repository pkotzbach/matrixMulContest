#pragma once
#include <iostream>
#include <assert.h>
#include <immintrin.h>
#include <thread>

//#define DEBUG
#define IN_SIMD 8

#define T_x 40
#define T_y 32
#define T_i 16

namespace MyAlgebra
{
    const int MEDIUM_MATRIX_BOUND = 192 * 192;
    const int BIG_MATRIX_BOUND = 455 * 455;

    const std::string author_name = "Pawel_Kotzbach";

    class CMtx
    {
    private:

        float** m_matrix;
        int     m_rows;
        int     m_columns;

        void deleteMatrix();
        void copyFrom(const CMtx& otherT);
        void moveFrom(CMtx&& otherT);

        inline bool equalSize(const CMtx& otherM) const { return m_columns == otherM.m_columns && m_rows == otherM.m_rows; }
        void fillWith(int val);
        void fillRandom();

        void mulScalarSIMD(float scalar, CMtx& result, int x1, int y1, int x2, int y2) const;
        void mulTilesSIMD(const CMtx& otherM, CMtx& result, int x1, int y1, int x2, int y2, const int N_mul) const;
        void addSIMD(const CMtx& otherM, CMtx& result, int x1, int y1, int x2, int y2) const;
        void subSIMD(const CMtx& otherM, CMtx& result, int x1, int y1, int x2, int y2) const;

        void mulBigMatrix(const CMtx& otherM, CMtx& result) const;
        void mulMediumMatrix(const CMtx& otherM, CMtx& result) const;
        void mulSmallMatrix(const CMtx& otherM, CMtx& result) const;
        void mulMatrixIKJ(const CMtx& otherM, CMtx& result) const;

        void createPrivates(int columns, int rows);

    public:
        static const float ALG_PRECISION;

        void set(int i, int j, float val) { m_matrix[i][j] = val; }

        int getRows() { return m_rows; }
        int getColumns() { return m_columns; }

        std::string authorName() { return author_name; }

        // =========================================================================
        // KONSTRUKTORY:
        // =========================================================================

        // Tworzy macierz z mozliwoscia losowej inicjalizacji
        CMtx(int columns, int rows, bool rand_init = false);

        // Tworzy kwadratowa macierz diagonalna
        CMtx( int row_cnt, float diagonal );

        CMtx(const CMtx& rhs);
        CMtx(CMtx&& otherM);

        ~CMtx();

        // =========================================================================
        // OPERATORY PRZYPISANIA:
        // =========================================================================

        //bylo const CMtx & wszedzie w returnie, zamienilem na void, moze byc tak?
        void operator=(const CMtx& rhs);

        // Zamiana macierzy na macierz diagonalna
        void operator=( const float diagonal );

        // Operator przenoszacy
        void operator=(CMtx&& rhs);


        // =========================================================================
        // INDEKSOWANIE MACIERZY
        // =========================================================================

        float* operator[](int row_ind);

        // =========================================================================
        // OPERACJE ALGEBRAICZNE
        // =========================================================================

        // Mnozenie macierzy przez wektor, rhs musi byc wektorem kolumnowym
//        CVct operator*( const CVct & rhs ) const;

        CMtx operator*(const CMtx& rhs) const;

        // Mnozenie macierzy przez stala
        CMtx operator*(float multiplier) const;

        CMtx operator+(const CMtx& rhs) const;
        CMtx operator-(const CMtx& rhs) const;

        // Minus unarny - zmiana znaku wszystkich wspoczynnikow macierzy
        CMtx operator-() const;

        // Transponowanie macierzy
        CMtx operator~() const;

        // Akceptuje tylko power >= -1:
        //    power = -1 - zwraca macierz odwrocona
        //    power = 0  - zwraca macierz jednostkowa
        //    power = 1  - zwraca kopie macierzy
        //    power > 1  - zwraca iloczyn macierzy
        CMtx operator^( int power ) const;

        // Porownywanie macierzy z dokladnoscia do stalej ALG_PRECISION
        bool operator==(const CMtx && rhs) const;

        // Tylko do celow testowych - wypisuje macierz wierszami na stdout
        void display() const;

        // friend CMtx operator*( FPTYPE multiplier, const CMtx &rhs );
    };

    CMtx operator*(float multiplier, const CMtx& rhs);
}


