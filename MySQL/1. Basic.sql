-- 1. <<Database commands>>
CREATE DATABASE test_db; 
USE test_db; 
DROP DATABASE test_1;  
ALTER DATABASE test_db READ ONLY = 1; 
ALTER DATABASE test_db READ ONLY = 0; 


-- 2. <<Tables commands>>
CREATE TABLE drug_info (
    DrugID INT, -- integer
    Chemical_name VARCHAR(255), -- short string 
    Product_Name VARCHAR(255),
    Dosage_form VARCHAR(255),
    Dosage VARCHAR(255), 
    Indication TEXT
    -- column decimal (3,2) -- decimal limited 3 digit, precise to 2 digit
);


SELECT * FROM drug_info;
RENAME TABLE drug_info_2 to drug_info;
DROP TABLE test_tb;

-- add column
ALTER TABLE drug_info
ADD Launch_date int;

-- rename column
ALTER TABLE drug_info
RENAME COLUMN Product_Name to Brand_name;

-- modify column type
ALTER TABLE drug_info
MODIFY COLUMN Launch_date date;

-- change column position
ALTER TABLE drug_info
MODIFY COLUMN Launch_date date
First;

ALTER TABLE drug_info
MODIFY Column Launch_date date
AFTER DrugID;

-- drop column
ALTER TABLE drug_info
DROP COLUMN column_name;


-- 3. <<Row commands>>
-- Add values to table/rows
DROP TABLE drug_info;
CREATE TABLE drug_info (
    DrugID INT, -- integer
    Launch_date DATE,
    Chemical_name VARCHAR(255), -- short string 
    Brand_name VARCHAR(255),
    Dosage_form VARCHAR(255),
    Dosage VARCHAR(255), 
    Indication TEXT
    -- column decimal (3,2) -- decimal limited 3 digit, precise to 2 digit
);


INSERT INTO drug_info
VALUES 
    (1, "1997-1-2", "Acetaminophen", "Actamin", "oral capsule", "500mg", "Pain reliever" ),
    (2, "2013-3-16", "Benzonatate", "Tessalon", "oral capsule", "100mg", "Cough reliver"),
    (3, "2002-12-30", "Adalimumab", "Humira", "subcutaneous injection (Pen)", "80mg/0.8ml", "Antirheumatic, TNF alfa inhibitors" ),
    (4, "2017-12-5", "Semaglutide", "Ozempic", "subcutaneous injection (Pen)", "0.25mg/0.5ml", "Antidiabetic, GLP-1 Agonist");
SELECT * FROM drug_info;

-- import CSV
-- make sure import and table can be alligned
SET GLOBAL LOCAL_INFILE=TRUE;
LOAD DATA LOCAL INFILE 'D:/MySQL_note/more_drug_data.csv'
    INTO TABLE drug_info
    FIELDS TERMINATED BY ','  
    ENCLOSED BY '"' 
    LINES TERMINATED BY '\n' 
    IGNORE 1 LINES;
    
-- INSERT non-duplicated data from CSV
CREATE TEMPORARY TABLE temp LIKE drug_info;

SET GLOBAL LOCAL_INFILE=TRUE;
LOAD DATA LOCAL INFILE 'D:/MySQL_note/full_drug_list.csv'
    INTO TABLE temp
    FIELDS TERMINATED BY ','  
    ENCLOSED BY '"' 
    LINES TERMINATED BY '\n' 
    IGNORE 1 LINES;

INSERT INTO drug_info 
SELECT *
FROM temp
WHERE NOT EXISTS(SELECT * 
             FROM drug_info
             WHERE (drug_info.DrugID = temp.DrugID ));
DROP TEMPORARY TABLE temp;	

-- Update multiple rows
set sql_safe_updates =0;
update drug_info
set project_manager = (case when drugid = 1 then 1001
                            when drugid = 2 then 1001
                            when drugid = 3 then 1003
                            when drugid = 4 then 1002
							when drugid = 5 then 1001
                            when drugid = 6 then 1002
                            when drugid = 7 then 1005
							when drugid = 8 then 1004
                            when drugid = 9 then 1001
                            when drugid = 10 then 1001
                            end);




-- select column(s) from table
SELECT * FROM drug_info;

SELECT DrugID, Brand_name from drug_info;

SELECT * FROM drug_info
WHERE DrugID = 1;
-- WHERE DrugID != 2;
-- WHERE indication IS NOT NULL;
-- WHERE indication IS NULL;
-- WHERE Launch_date > "2001-01-01";

-- Update/delete table
SET SQL_SAFE_UPDATES = 0;
UPDATE drug_info
SET Launch_date = "2002-12-29",
	-- SET Launch_date = NULL
	Indication = "Antirheumatic, TNF alpha inhibitors"
WHERE DrugID = 3;
SELECT * FROM drug_info;

DELETE FROM drug_info_2
WHERE DrugID = 3;

-- Date/time/datatime 
CREATE TABLE date_time(
    date DATE,
    time TIME,
    datetime DATETIME);

INSERT INTO date_time
VALUES(current_date(), current_time(), now());
-- VALUES(current_date() + 1, current_time(), now());
SELECT * FROM date_time;


-- Commit/ROLLBACK
COMMIT;
SET AUTOCOMMIT = 0;
DELETE FROM drug_info
WHERE DrugID = 1;
ROLLBACK;
SELECT * FROM drug_info;
SET AUTOCOMMIT = 1;
