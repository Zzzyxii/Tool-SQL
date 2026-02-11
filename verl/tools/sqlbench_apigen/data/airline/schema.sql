PRAGMA foreign_keys = ON;

CREATE TABLE airports (
            code TEXT PRIMARY KEY,
            city TEXT,
            state TEXT,
            country TEXT
        );

CREATE TABLE flight_schedules (
            flight_number TEXT NOT NULL,
            departure_date TEXT NOT NULL,
            status TEXT,
            actual_departure_time_est TEXT,
            actual_arrival_time_est TEXT,
            available_basic_economy INTEGER,
            available_economy INTEGER,
            available_business INTEGER,
            price_basic_economy REAL,
            price_economy REAL,
            price_business REAL,
            PRIMARY KEY (flight_number, departure_date),
            FOREIGN KEY (flight_number) REFERENCES flights(flight_number) ON DELETE CASCADE
        );

CREATE TABLE flights (
            flight_number TEXT PRIMARY KEY,
            origin TEXT,
            destination TEXT,
            scheduled_departure_time_est TEXT,
            scheduled_arrival_time_est TEXT
        );

...