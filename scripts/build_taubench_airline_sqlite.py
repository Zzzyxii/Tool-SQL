import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime

# Paths
DATA_DIR = Path("verl/tools/taubench_airline/data")
SCHEMA_PATH = Path("verl/tools/sqlbench_apigen/data/airline/schema.sql")
OUTPUT_DB_PATH = Path("verl/tools/sqlbench_apigen/data/airline/airline_new.sqlite")

# Airport Data (Hardcoded from list_all_airports.py + manual state/country)
AIRPORTS = [
    ("SFO", "San Francisco", "CA", "USA"),
    ("JFK", "New York", "NY", "USA"),
    ("LAX", "Los Angeles", "CA", "USA"),
    ("ORD", "Chicago", "IL", "USA"),
    ("DFW", "Dallas", "TX", "USA"),
    ("DEN", "Denver", "CO", "USA"),
    ("SEA", "Seattle", "WA", "USA"),
    ("ATL", "Atlanta", "GA", "USA"),
    ("MIA", "Miami", "FL", "USA"),
    ("BOS", "Boston", "MA", "USA"),
    ("PHX", "Phoenix", "AZ", "USA"),
    ("IAH", "Houston", "TX", "USA"),
    ("LAS", "Las Vegas", "NV", "USA"),
    ("MCO", "Orlando", "FL", "USA"),
    ("EWR", "Newark", "NJ", "USA"),
    ("CLT", "Charlotte", "NC", "USA"),
    ("MSP", "Minneapolis", "MN", "USA"),
    ("DTW", "Detroit", "MI", "USA"),
    ("PHL", "Philadelphia", "PA", "USA"),
    ("LGA", "New York", "NY", "USA"),
]

def load_json(filename):
    with open(DATA_DIR / filename, 'r') as f:
        return json.load(f)

def create_db():
    if OUTPUT_DB_PATH.exists():
        os.remove(OUTPUT_DB_PATH)
    
    conn = sqlite3.connect(OUTPUT_DB_PATH)
    cursor = conn.cursor()
    
    # Load Schema
    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()
        cursor.executescript(schema_sql)
    
    return conn

def populate_airports(conn):
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO airports (code, city, state, country) VALUES (?, ?, ?, ?)",
        AIRPORTS
    )
    print(f"Inserted {len(AIRPORTS)} airports.")

def populate_users(conn, users_data):
    cursor = conn.cursor()
    users_rows = []
    payment_rows = []
    saved_passenger_rows = []
    
    for user_id, user in users_data.items():
        # Users table
        users_rows.append((
            user_id,
            user["name"]["first_name"],
            user["name"]["last_name"],
            user["email"],
            user.get("membership"), # Might be missing
            user["dob"],
            user["address"]["address1"],
            user["address"]["address2"],
            user["address"]["city"],
            user["address"]["state"],
            user["address"]["country"],
            user["address"]["zip"]
        ))
        
        # Payment Methods
        for pm_id, pm in user.get("payment_methods", {}).items():
            payment_rows.append((
                user_id,
                pm_id,
                pm.get("source"),
                pm.get("brand"),
                pm.get("last_four"),
                None # amount is usually for certificates, not cards in this source?
            ))
            
        # Saved Passengers
        for idx, sp in enumerate(user.get("saved_passengers", [])):
            saved_passenger_rows.append((
                user_id,
                idx,
                sp["first_name"],
                sp["last_name"],
                sp["dob"]
            ))

    cursor.executemany(
        "INSERT INTO users (user_id, first_name, last_name, email, membership, dob, address1, address2, city, state, country, zip) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        users_rows
    )
    cursor.executemany(
        "INSERT INTO user_payment_methods (user_id, payment_id, source, brand, last_four, amount) VALUES (?, ?, ?, ?, ?, ?)",
        payment_rows
    )
    cursor.executemany(
        "INSERT INTO user_saved_passengers (user_id, passenger_index, first_name, last_name, dob) VALUES (?, ?, ?, ?, ?)",
        saved_passenger_rows
    )
    print(f"Inserted {len(users_rows)} users.")

def populate_flights(conn, flights_data):
    cursor = conn.cursor()
    flights_rows = []
    schedule_rows = []
    
    for flight_num, flight in flights_data.items():
        # Flights table
        flights_rows.append((
            flight_num,
            flight["origin"],
            flight["destination"],
            flight["scheduled_departure_time_est"],
            flight["scheduled_arrival_time_est"]
        ))
        
        # Flight Schedules
        for date_str, date_info in flight.get("dates", {}).items():
            schedule_rows.append((
                flight_num,
                date_str,
                date_info.get("status"),
                date_info.get("actual_departure_time_est"),
                date_info.get("actual_arrival_time_est"),
                date_info.get("available_basic_economy"),
                date_info.get("available_economy"),
                date_info.get("available_business"),
                date_info.get("price_basic_economy"),
                date_info.get("price_economy"),
                date_info.get("price_business")
            ))
            
    cursor.executemany(
        "INSERT INTO flights (flight_number, origin, destination, scheduled_departure_time_est, scheduled_arrival_time_est) VALUES (?, ?, ?, ?, ?)",
        flights_rows
    )
    cursor.executemany(
        "INSERT INTO flight_schedules (flight_number, departure_date, status, actual_departure_time_est, actual_arrival_time_est, available_basic_economy, available_economy, available_business, price_basic_economy, price_economy, price_business) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        schedule_rows
    )
    print(f"Inserted {len(flights_rows)} flights and {len(schedule_rows)} schedules.")

def populate_reservations(conn, reservations_data):
    cursor = conn.cursor()
    res_rows = []
    res_flight_rows = []
    res_passenger_rows = []
    res_payment_rows = []
    
    for res_id, res in reservations_data.items():
        # Reservations table
        res_rows.append((
            res_id,
            res["user_id"],
            res["origin"],
            res["destination"],
            res["flight_type"],
            res["cabin"],
            res["total_baggages"],
            res["nonfree_baggages"],
            res["insurance"],
            res.get("status", "confirmed")
        ))
        
        # Reservation Flights
        for idx, flight in enumerate(res.get("flights", [])):
            res_flight_rows.append((
                res_id,
                idx,
                flight["flight_number"],
                flight["date"],
                flight["origin"],
                flight["destination"],
                flight["price"]
            ))
            
        # Reservation Passengers
        for idx, pax in enumerate(res.get("passengers", [])):
            res_passenger_rows.append((
                res_id,
                idx,
                pax["first_name"],
                pax["last_name"],
                pax["dob"]
            ))
            
        # Reservation Payments
        for idx, pay in enumerate(res.get("payment_methods", [])):
            res_payment_rows.append((
                res_id,
                idx,
                pay["payment_id"],
                pay["amount"]
            ))

    cursor.executemany(
        "INSERT INTO reservations (reservation_id, user_id, origin, destination, flight_type, cabin, total_baggages, nonfree_baggages, insurance, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        res_rows
    )
    cursor.executemany(
        "INSERT INTO reservation_flights (reservation_id, segment_index, flight_number, flight_date, origin, destination, price) VALUES (?, ?, ?, ?, ?, ?, ?)",
        res_flight_rows
    )
    cursor.executemany(
        "INSERT INTO reservation_passengers (reservation_id, passenger_index, first_name, last_name, dob) VALUES (?, ?, ?, ?, ?)",
        res_passenger_rows
    )
    cursor.executemany(
        "INSERT INTO reservation_payments (reservation_id, payment_index, payment_id, amount) VALUES (?, ?, ?, ?)",
        res_payment_rows
    )
    print(f"Inserted {len(res_rows)} reservations.")

def main():
    print("Loading JSON data...")
    users = load_json("users.json")
    flights = load_json("flights.json")
    reservations = load_json("reservations.json")
    
    print("Creating database...")
    conn = create_db()
    
    print("Populating tables...")
    populate_airports(conn)
    populate_users(conn, users)
    populate_flights(conn, flights)
    populate_reservations(conn, reservations)
    
    conn.commit()
    conn.close()
    print(f"Database created at {OUTPUT_DB_PATH}")

if __name__ == "__main__":
    main()
