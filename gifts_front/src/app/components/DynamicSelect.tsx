import React from "react";
import { useController, Control } from "react-hook-form";
import {
    FormControl,
    FormItem,
    FormLabel,
    FormMessage,
} from "@/components/ui/form";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";

interface DynamicSelectProps {
    control: Control;
    name: string;
    label: string;
    options: string[];
    placeholder: string;
}

const DynamicSelect: React.FC<DynamicSelectProps> = ({ control, name, label, options, placeholder }) => {
    const {
        field,
        fieldState: { error },
    } = useController({
        name,
        control,
        defaultValue: "",
        rules: { required: "Please select an option" }, // Add validation rule
    });

    return (
        <FormItem className="w-1/4 m-2">
            <FormLabel>{label}</FormLabel>
            <Select onValueChange={field.onChange} defaultValue={field.value}>
                <FormControl>
                    <SelectTrigger>
                        <SelectValue placeholder={placeholder} />
                    </SelectTrigger>
                </FormControl>
                <SelectContent>
                    {options.map((option, index) => (
                        <SelectItem key={index} value={option}>
                            {option}
                        </SelectItem>
                    ))}
                </SelectContent>
            </Select>
            <FormMessage>{error ? error.message : null}</FormMessage>
        </FormItem>
    );
};

export default DynamicSelect;
