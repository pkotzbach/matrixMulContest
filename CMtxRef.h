#pragma once
#include <iostream>
#include <assert.h>
#include <immintrin.h>
#include <thread>

#define DEBUG 0

namespace MyRefAlgebra
{
    class CVct;

    class CMtx
    {
    protected:
        float** m_matrix; //bylo FPTYPE, powinien byc float?
        int     m_rows;
        int     m_columns;

        void deleteMatrix();
        void copyFrom(const CMtx& otherT);
        void moveFrom(CMtx&& otherT);

        inline bool equalSize(const CMtx& otherM) const { return m_columns == otherM.m_columns && m_rows == otherM.m_rows; }

    public:
        static const float ALG_PRECISION;

        int getRows() { return m_rows; }
        int getColumns() { return m_columns; }

        void set(int x, int y, float val) { m_matrix[x][y] = val; }

        // =========================================================================
        // KONSTRUKTORY:
        // =========================================================================

        // Tworzy macierz z mozliwoscia losowej inicjalizacji
        CMtx(int columns, int rows, bool rand_init = false);

        // Tworzy kwadratowa macierz diagonalna
//        CMtx( int row_cnt, float diagonal );

        CMtx(const CMtx& rhs);
        CMtx(CMtx&& otherM);

        // Jesli potrzeba - nalezy zadeklarowac i zaimplementowac inne konstruktory
        ~CMtx();


        // =========================================================================
        // OPERATORY PRZYPISANIA:
        // =========================================================================

        //bylo const CMtx & wszedzie w returnie, zamienilem na void, moze byc tak?
        void operator=(const CMtx& rhs);

        // Zamiana macierzy na macierz diagonalna
//        void operator=( const float diagonal );

        // Operator pzzenoszacy
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
//        CMtx operator^( int power ) const;

        // Porownywanie macierzy z dokladnoscia do stalej ALG_PRECISION
//        bool operator==(const CMtx && rhs) const;

        // Tylko do celow testowych - wypisuje macierz wierszami na stdout
        void display() const;

        // friend CMtx operator*( FPTYPE multiplier, const CMtx &rhs );
    };

//    CMtx operator*(float multiplier, const CMtx &rhs);
}


