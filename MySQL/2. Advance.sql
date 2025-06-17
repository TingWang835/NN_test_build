-- <<Constraint>>
DROP TABLE drug_info;
CREATE TABLE drug_info (
    DrugID INT, -- integer
    Launch_date DATE,
    Chemical_name VARCHAR(255), -- short string 
    Brand_name VARCHAR(255),
    Dosage_form VARCHAR(255),
    Dosage VARCHAR(255), 
    Indication TEXT,
    CONSTRAINT pk_drugid PRIMARY KEY (DrugID)
);


SET GLOBAL LOCAL_INFILE=TRUE;
    LOAD DATA LOCAL INFILE 'D:/MySQL_note/full_drug_list.csv'
    INTO TABLE drug_info
    FIELDS TERMINATED BY ','  
    ENCLOSED BY '"' 
    LINES TERMINATED BY '\n' 
    IGNORE 1 LINES;

CREATE TABLE sales (
    DrugID INT AUTO_INCREMENT, 
    Price DECIMAL(10,2) DEFAULT 0, -- set default = 0
    Sold INT,
    Unit VARCHAR (255) DEFAULT ('package'), -- set default as package
    Total_Sale DECIMAL(10,2),
    CONSTRAINT pk_drugid PRIMARY KEY (DrugID), -- limited 1 pk/table, can pk multiple cols
    CONSTRAINT chk_total_sale CHECK (Total_sale >= Price), -- check if condition is met
    CONSTRAINT fk_drug_info_id FOREIGN KEY (DrugID) REFERENCES drug_info(DrugID) -- fk limits new insert if not recongnized on reference table, i.e. setting fk on both tables prohibits any new insert
    ON DELETE SET NULL
	-- CONSTRAINT uc_drugid UNIQUE (DrugID, Chemical_name),
    -- CONSTRAINT chk_not_null CHECK (DrugID != 0), -- = NOT NULL
);

INSERT INTO sales (Price, Sold)
VALUES (25.6, 58620),
       (63.7, 106546),
       (66.5, 156283),
       (536.8, 185321),
       (85.9, 175269),
       (85.2, 16348),
       (718.2, 124563),
       (125.7, 175634),
       (17.2, 556934),
       (12.3, 446982);
SELECT * from sales;

CREATE TABLE empolyees (
    Empolyee_id INT UNIQUE AUTO_INCREMENT,
    First_name VARCHAR(255),
    Last_name VARCHAR(255),
    Job VARCHAR(255),
    Hourly_pay DECIMAL(3,2),
    Hire_date DATE,
    Manager INT);
ALTER TABLE empolyees AUTO_INCREMENT = 1000;

INSERT INTO empolyees (first_name, last_name, job, hourly_pay, hire_date, manager)
VALUES ("Adam", "Smith", "Department Manager", 100, "2013-01-08", NULL),
       ("David", "Finch", "Chief Researcher", 80, "2013-01-09", 1000), 
       ("Samatha", "Cody", "Researcher I", 65, "2013-02-06", 1001), 
       ("John", "Smith", "Researcher I", 60, "2016-06-09", 1001), 
       ("Mary", "James", "Researcher II", 45, "2017-08-22", 1002),
       ("Jon", "Dean", "Researcher III", 30, "2018-04-30", 1004),
       ("Janice", "Alice", "Researcher II", 40, "2019-05-18", 1003),
       ("Jacob", "Smith", "Researcher III", 25, "2020-11-10", 1006);
SELECT * from empolyees;

ALTER TABLE drug_info
ADD COLUMN project_manager INT;
    



-- add/remove PRIMARY KEY, UNIQUE, CHECK
ALTER TABLE sales
ADD CONSTRAINT pk_drugid PRIMARY KEY (DrugID); -- or (DrugID, Chemical_name)
ALTER TABLE sales
DROP CONSTRAINT pk_drugid;


-- add/remove FOREIGN KEY
ALTER TABLE sales
ADD CONSTRAINT fk_drug_info_id
FOREIGN KEY (DrugID) REFERENCES drug_info(DrugID);
-- reference must be a primary key
ALTER TABLE sales
DROP CONSTRAINT fk_link;

-- ON DELETE
DELETE FROM sales
WHERE DrugID = 5;
-- drugid = 5 can not be deleted bc foreign key 
-- to delete it:
-- SET foreign_key_checks = 0

-- ON DELETE allows more option when deleting data bound by fk
-- Create table with on delete 
CREATE TABLE drug_info (
    DrugID INT, -- integer
    -- etc
    CONSTRAINT fk_sales_id FOREIGN KEY (DrugID) REFERENCES sales(DrugID)
    ON DELETE SET NULL  -- on delete cascade -- delete everything linked by the fk
);

-- add/remove ON DELETE
ALTER TABLE sales DROP FOREIGN KEY fk_drug_info_id;
ALTER TABLE sales
ADD CONSTRAINT fk_drug_info_id
FOREIGN KEY (DrugID) REFERENCES drug_info(DrugID)
ON DELETE CASCADE; -- ON DELETE SET NULL

-- add/remove NOT NULL
ALTER TABLE sales MODIFY DrugID INT NOT NULL; 
ALTER TABLE sales MODIFY DrugID INT NULL;

-- add/remove DEFAULT
ALTER TABLE sales ALTER Unit SET DEFAULT 'package';
ALTER TABLE sales ALTER Unit DROP DEFAULT;

-- alter/remove AUTO_INCREMENT
ALTER TABLE sales AUTO_INCREMENT = 100; -- starting from 100
ALTER TABLE sales 
MODIFY DrugID INT,
DROP PRIMARY KEY;

-- <<Joins>>
-- Inner/Left/Right join
SELECT drug_info.drugid, chemical_name, brand_name, Dosage, Price, Sold
FROM drug_info INNER JOIN sales -- show rows shared by both tables
-- FROM drug_info LEFT JOIN sales -- show all rows from left tables
-- FROM drug_info RIGHT JOIN sales -- show all rows form right table
ON drug_info.DrugID = sales.DrugID;

-- Full/Outer Join (for MySQL)
SELECT drug_info.drugid, chemical_name, brand_name, Dosage, Price, Sold
FROM drug_info LEFT JOIN sales 
ON drug_info.DrugID = sales.DrugID
UNION
SELECT drug_info.drugid, chemical_name, brand_name, Dosage, Price, Sold -- selects have to match
FROM drug_info RIGHT JOIN sales 
ON drug_info.DrugID = sales.DrugID;

-- Self Join
SELECT a.empolyee_id, a.first_name, a.last_name, 
concat(b.first_name, " ", b.last_name) AS "Reports_to"
FROM empolyees AS a
LEFT JOIN empolyees AS b
ON a.manager = b.Empolyee_id;

-- <<Functions>>
-- Count
SELECT COUNT(chemical_name) AS "Oral Capsules"
FROM drug_info
WHERE dosage_form = "oral capsule";

-- Min/Max/AVG/SUM
SELECT MIN(price) AS "Min Price"
-- SELECT MAX(price)
-- SELECT AVG(price)
-- SELECT SUM(price)
FROM sales;

-- CONCAT
SELECT CONCAT(chemical_name, " (", brand_name, ")") AS "Product"
FROM drug_info;

-- <<Logical Operators>>
-- AND/OR/NOT
SELECT * FROM drug_info
WHERE launch_date < "2000-1-1" AND dosage_form = "oral capsule";
-- WHERE launch_date < "2000-1-1" OR dosage_form = "oral capsule";
-- WHERE NOT launch_date < "2000-1-1" AND NOT dosage_form = "oral capsule";

-- BETWEEN/IN
SELECT * FROM drug_info
WHERE launch_date BETWEEN "2010-1-1" AND "2025-1-1";

SELECT * FROM drug_info
WHERE dosage_form IN ("extended-release capsule", "oral capsule", "oral suspension");

-- Wild card characters (_ and %)
SELECT * FROM drug_info
WHERE chemical_name LIKE "ben%";
-- WHERE chemical_name LIKE "%phen";

SELECT * FROM drug_info
WHERE launch_date LIKE "____-01-__" ;

-- <<Clause>>
-- ORDER BY
SELECT * FROM drug_info
ORDER BY chemical_name DESC; -- or ASC for ascending

SELECT * FROM drug_info -- to order by multiple columns
ORDER BY chemical_name DESC, launch_date;

-- limit/ pagination
SELECT * FROM drug_info
LIMIT 2, 3; -- 1st number = offset (skip # rows), 2nd number show # rows
-- or 
-- LIMIT 4 OFFSET 4;

-- combine order and limit
SELECT * FROM drug_info
ORDER BY brand_name DESC LIMIT 3;

-- <<Miscellaneous>>
-- UNION
SELECT chemical_name, brand_name FROM drug_info
UNION
SELECT price, sold FROM sales; -- to UNION 2 tables, they must select the same number of columns (data type dont matter)

SELECT chemical_name, brand_name FROM drug_info
UNION ALL -- UNION ALL allows duplicates
SELECT price, sold FROM sales;

-- VIEW
select * from empolyees;
CREATE VIEW Report_to AS
SELECT a.empolyee_id, a.first_name, a.last_name, a.job,
concat(b.first_name, " ", b.last_name) AS "Reports_to"
FROM empolyees AS a
LEFT JOIN empolyees AS b
ON a.manager = b.Empolyee_id;
SELECT * FROM report_to;


-- INDEX
-- Index (BTree data structure) make SELECT faster but UPDATE slower
CREATE INDEX last_name_idx
ON empolyees (last_name);
SHOW INDEXES FROM empolyees;

CREATE INDEX full_name_idx 
ON empolyees (last_name, first_name); -- sequence determined priority
SHOW INDEXES FROM empolyees;
-- if SELECT by last_name/ Last AND first_name , full_name_idx is active;
-- SELECT only by first_name, full_name_idx is not active

ALTER TABLE empolyees -- to remove an index
DROP INDEX last_name_idx;

-- Subqueries
SELECT AVG (Price) FROM sales;
SELECT drugid, price, (SELECT AVG (Price) FROM sales) AS AVG_Price -- put a query in braket making it a subquery
FROM sales;
SELECT drugid, price, (SELECT AVG (Price) FROM sales) AS AVG_Price
FROM sales
WHERE price <= (SELECT AVG (Price) FROM sales);

SELECT drug_info.drugid, chemical_name, brand_name, Dosage, Price, Sold, (SELECT AVG (Price) FROM sales) AS AVG_Price
FROM drug_info INNER JOIN sales 
ON drug_info.DrugID = sales.DrugID
WHERE price > (SELECT AVG (Price) FROM sales);

-- Group by/ROLLUP
SELECT AVG(hourly_pay), job -- count avg min max sum
FROM empolyees
GROUP BY job
WITH ROLLUP; -- add extra row for rollup

SELECT AVG(hourly_pay), job
FROM empolyees
GROUP BY job
WITH ROLLUP -- ROLLUP ignores HAVING and includes all rows
HAVING AVG(Hourly_pay) >= (SELECT AVG(Hourly_pay) FROM empolyees);
-- HAVING = WHERE (WHERE can not be used in group by)

SELECT AVG(hourly_pay), DATE_FORMAT(hire_date, '%Y') AS Hire_year
FROM empolyees
GROUP BY DATE_FORMAT(hire_date, '%Y') -- '%M', '%D', "%Y%M' etc
WITH ROLLUP; 

-- Stored Procedures (creating shortcut for code/statement)
delimiter // 
CREATE PROCEDURE show_manager()
BEGIN
	SELECT a.empolyee_id, a.first_name, a.last_name, 
	concat(b.first_name, " ", b.last_name) AS "Reports_to"
	FROM empolyees AS a
    LEFT JOIN empolyees AS b
    ON a.manager = b.Empolyee_id;
END
//
delimiter ;

-- Call/ drop dprocedure
CALL show_manager();
DROP PROCEDURE show_manager;

-- Procedure with parameter
delimiter // 
CREATE PROCEDURE find_drug(IN chem VARCHAR(255),
						   IN brand VARCHAR(255))
BEGIN
	SELECT * FROM drug_info
    WHERE chemical_name = chem
    OR Brand_name = brand;
END
//
delimiter ;
drop procedure find_drug;
CALL find_drug(null, "Ozempic");
CALL find_drug("Adalimumab", null);

-- Trigger
CREATE TRIGGER before_total_sale_update
BEFORE UPDATE ON sales
FOR EACH ROW
SET NEW.total_sale = NEW.price * NEW.sold;
SHOW TRIGGERS;

UPDATE sales
SET Price = 25.7
WHERE DrugID = 1; -- only activate trigger for the updated one

set sql_safe_updates = 0;
UPDATE sales
SET Price = Price + 0.09; 
set sql_safe_updates = 1;


CREATE TRIGGER before_total_sale_insert
BEFORE INSERT ON sales
FOR EACH ROW
SET NEW.total_sale = NEW.price * NEW.sold;

INSERT INTO sales (DrugID, Price, Sold)
VALUES (11, 45.99, 1000);

INSERT INTO drug_info (DrugID, chemical_name, brand_name)
VALUES (11, "a", "b");

-- Trigger between tables
CREATE TABLE balance (
    balance_id INT PRIMARY KEY AUTO_INCREMENT,
    balance_type VARCHAR(255),
    amount DECIMAL(10,2)
);
SELECT * FROM balance;

INSERT INTO balance (balance_type)
values ("salaries"), ("profits");

CREATE TRIGGER before_salaryies_update
BEFORE UPDATE ON empolyees
FOR EACH ROW
UPDATE balance
SET amount = (SELECT SUM(hourly_pay) FROM empolyees) * 2080 * -1
WHERE balance_id = 1;

DROP TRIGGER after_salaryies_update;